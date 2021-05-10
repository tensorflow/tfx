# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilies for user defined functions.

TFX-internal use only and experimental, no backwards compatibilty guarantees.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

from typing import Any, Callable, Dict, List, Optional, Text, Tuple

from absl import logging

from tfx.dsl.components.base import base_component
from tfx.dsl.io import fileio
from tfx.utils import import_utils
from tfx.utils import io_utils

# Key for module file path.
_MODULE_FILE_KEY = 'module_file'
# Key for module python path.
_MODULE_PATH_KEY = 'module_path'
# Ephemeral setup.py file name.
_EPHEMERAL_SETUP_PY_FILE_NAME = '_tfx_generated_setup.py'


# TODO(b/157155972): improve user code support.
def get_fn(exec_properties: Dict[Text, Any],
           fn_name: Text) -> Callable[..., Any]:
  """Loads and returns user-defined function."""
  logging.error('udf_utils.get_fn %r %r', exec_properties, fn_name)

  has_module_file = bool(exec_properties.get(_MODULE_FILE_KEY))
  has_module_path = bool(exec_properties.get(_MODULE_PATH_KEY))
  has_fn = bool(exec_properties.get(fn_name))

  if has_module_path:
    module_path = exec_properties[_MODULE_PATH_KEY]
    return import_utils.import_func_from_module(module_path, fn_name)
  elif has_module_file:
    if has_fn:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], exec_properties[fn_name])
    else:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], fn_name)
  elif has_fn:
    fn_path_split = exec_properties[fn_name].split('.')
    return import_utils.import_func_from_module('.'.join(fn_path_split[0:-1]),
                                                fn_path_split[-1])
  else:
    raise ValueError(
        'Neither module file or user function have been supplied in `exec_properties`.'
    )


def try_get_fn(exec_properties: Dict[Text, Any],
               fn_name: Text) -> Optional[Callable[..., Any]]:
  """Loads and returns user-defined function if exists."""
  try:
    return get_fn(exec_properties, fn_name)
  except (ValueError, AttributeError):
    # ValueError: module file or user function is unset.
    # AttributeError: the function doesn't exist in the module.
    return None


def _get_ephemeral_setup_py_contents(package_name: Text, version_string: Text,
                                     module_names: List[Text]):
  return f"""import setuptools

setuptools.setup(
    name={repr(package_name)},
    version={repr(version_string)},
    author='TFX User',
    author_email='nobody@example.com',
    description='Auto-generated TFX user code package.',
    py_modules={repr(module_names)},
    classifiers=[],
    python_requires='>=3.6',
)
"""


def should_package_user_modules():
  """Whether to package user modules in the current execution environment."""
  if os.environ.get('UNSUPPORTED_DO_NOT_PACKAGE_USER_MODULES'):
    return False
  return True


class UserModuleFilePipDependency(base_component._PipDependencyFuture):  # pylint: disable=protected-access
  """Specification of a user module dependency."""

  def __init__(self, component: base_component.BaseComponent,
               module_file_key: Text, module_path_key: Text):
    self.component = component
    self.module_file_key = module_file_key
    self.module_path_key = module_path_key

  def resolve(self, pipeline_root: Text):
    # Package the given user module file as a Python wheel.
    module_file = self.component.spec.exec_properties[self.module_file_key]

    # Perform validation on the given `module_file`.
    if not module_file:
      return None
    elif not isinstance(module_file, Text):
      # TODO(b/187753042): Deprecate and remove usage of RuntimeParameters for
      # `module_file` parameters and remove this code path.
      logging.warning(
          'Module file %r for component %s is not a path string; '
          'skipping Python user module wheel packaging.', module_file,
          self.component)
      return None
    elif not fileio.exists(module_file):
      raise ValueError(
          'Specified module file %r for component %s does not exist.' %
          (module_file, self.component))

    # Perform validation on the `pipeline_root`.
    if not pipeline_root:
      logging.warning(
          'No pipeline root provided; skipping Python user module '
          'wheel packaging for component %s.', self.component)
      return None
    pipeline_root_exists = fileio.exists(pipeline_root)
    if not pipeline_root_exists:
      fileio.makedirs(pipeline_root)

    # Perform packaging of the user module.
    dist_file_path, user_module_path = package_user_module_file(
        self.component.id, module_file, pipeline_root)

    # Set the user module key to point to a module in this wheel, and clear the
    # module path key before returning.
    self.component.spec.exec_properties[self.module_path_key] = user_module_path
    self.component.spec.exec_properties[self.module_file_key] = None
    return dist_file_path


def add_user_module_dependency(component: base_component.BaseComponent,
                               module_file_key: Text,
                               module_path_key: Text) -> None:
  """Adds a module file dependency to the current component."""
  dependency = UserModuleFilePipDependency(component, module_file_key,
                                           module_path_key)
  component._add_pip_dependency(dependency)  # pylint: disable=protected-access


def _get_version_hash(user_module_dir: Text, source_files: List[Text]) -> Text:
  """Compute a version hash based on user module directory contents."""
  source_files = sorted(source_files)
  h = hashlib.sha256()
  for source_file in source_files:
    source_file_name_bytes = source_file.encode('utf-8')
    h.update(struct.pack('>Q', len(source_file_name_bytes)))
    h.update(source_file_name_bytes)
    with open(os.path.join(user_module_dir, source_file), 'rb') as f:
      file_contents = f.read()
    h.update(struct.pack('>Q', len(file_contents)))
    h.update(file_contents)
  return h.hexdigest()


def package_user_module_file(instance_name: Text, module_path: Text,
                             pipeline_root: Text) -> Tuple[Text, Text]:
  """Package the given user module file into a Python Wheel package.

  Args:
      instance_name: Name of the component instance, for creating a unique wheel
        package name.
      module_path: Path to the module file to be packaged.
      pipeline_root: Text

  Returns:
      dist_file_path: Path to the generated wheel file.
      user_module_path: Path for referencing the user module when stored
        as the _MODULE_PATH_KEY execution property. Format should be treated
        as opaque by the user.

  Raises:
      RuntimeError: When wheel building fails.
  """
  module_path = os.path.abspath(io_utils.ensure_local(module_path))
  if not module_path.endswith('.py'):
    raise ValueError('Module path %r is not a ".py" file.' % module_path)
  if not os.path.exists(module_path):
    raise ValueError('Module path %r does not exist.' % module_path)

  user_module_dir, module_file_name = os.path.split(module_path)
  user_module_name = re.sub(r'\.py$', '', module_file_name)
  source_files = []

  # Discover all Python source files in this directory for inclusion.
  for file_name in os.listdir(user_module_dir):
    if file_name.endswith('.py'):
      source_files.append(file_name)
  module_names = []
  for file_name in source_files:
    if file_name in (_EPHEMERAL_SETUP_PY_FILE_NAME, '__init__.py'):
      continue
    module_name = re.sub(r'\.py$', '', file_name)
    module_names.append(module_name)

  # Set up build directory.
  build_dir = tempfile.mkdtemp()
  for source_file in source_files:
    shutil.copyfile(
        os.path.join(user_module_dir, source_file),
        os.path.join(build_dir, source_file))

  # Generate an ephemeral wheel for this module.
  logging.info(
      'Generating ephemeral wheel package for %r (including modules: %s).',
      module_path, module_names)

  version_hash = _get_version_hash(user_module_dir, source_files)
  logging.info('User module package has hash fingerprint version %s.',
               version_hash)

  setup_py_path = os.path.join(build_dir, _EPHEMERAL_SETUP_PY_FILE_NAME)
  with open(setup_py_path, 'w') as f:
    f.write(
        _get_ephemeral_setup_py_contents('tfx-user-code-%s' % instance_name,
                                         '0.0+%s' % version_hash, module_names))

  temp_dir = tempfile.mkdtemp()
  dist_dir = tempfile.mkdtemp()
  bdist_command = [
      sys.executable, setup_py_path, 'bdist_wheel', '--bdist-dir', temp_dir,
      '--dist-dir', dist_dir
  ]
  logging.info('Executing: %s', bdist_command)
  try:
    subprocess.check_call(bdist_command, cwd=build_dir)
  except subprocess.CalledProcessError as e:
    raise RuntimeError('Failed to build wheel.') from e

  dist_files = os.listdir(dist_dir)
  if len(dist_files) != 1:
    raise RuntimeError(
        'Unexpectedly found %d output files in wheel output directory %s.' %
        (len(dist_files), dist_dir))
  build_dist_file_path = os.path.join(dist_dir, dist_files[0])
  # Copy wheel file atomically to wheel staging directory.
  dist_wheel_directory = os.path.join(pipeline_root, '_wheels')
  dist_file_path = os.path.join(dist_wheel_directory, dist_files[0])
  temp_dist_file_path = dist_file_path + '.tmp'
  fileio.makedirs(dist_wheel_directory)
  fileio.copy(build_dist_file_path, temp_dist_file_path, overwrite=True)
  fileio.rename(temp_dist_file_path, dist_file_path, overwrite=True)
  logging.info(
      ('Successfully built user code wheel distribution at %r; target user '
       'module is %r.'), dist_file_path, user_module_name)

  # Encode the user module key as a specification of a user module name within
  # a packaged wheel path.
  assert '@' not in user_module_name, ('Unexpected invalid module name: %s' %
                                       user_module_name)
  user_module_path = '%s@%s' % (user_module_name, dist_file_path)
  logging.info('Full user module path is %r', user_module_path)

  return dist_file_path, user_module_path


def decode_user_module_key(user_module_key: Text) -> Tuple[Text, List[Text]]:
  """Decode the given user module key into module path and pip dependencies."""
  if user_module_key and '@' in user_module_key:
    user_module_name, dist_file_path = user_module_key.split('@', maxsplit=1)
    return user_module_name, [dist_file_path]
  else:
    return user_module_key, []


class TempPipInstallContext:
  """Context manager for wrapped code and subprocesses to use pip package."""

  def __init__(self, pip_dependencies: List[Text]):
    if not isinstance(pip_dependencies, list):
      raise ValueError('Expected list of dependencies, got %r instead.' %
                       (pip_dependencies,))
    self.pip_dependencies = pip_dependencies
    self.temp_directory = None

  def __enter__(self) -> 'TempPipInstallContext':
    if self.pip_dependencies:
      self.temp_directory = tempfile.mkdtemp()
      for dependency in self.pip_dependencies:
        install_to_temp_directory(dependency, temp_dir=self.temp_directory)
      sys.path = sys.path + [self.temp_directory]
      os.environ['PYTHONPATH'] = ':'.join(sys.path)
    return self

  def __exit__(self, *unused_exc_info):
    if self.pip_dependencies:
      sys.path = list(path for path in sys.path if path != self.temp_directory)
      os.environ['PYTHONPATH'] = ':'.join(sys.path)


def install_to_temp_directory(pip_dependency: Text,
                              temp_dir: Optional[Text] = None) -> Text:
  """Install the given pip dependency specifier to a temporary directory.

  Args:
      pip_dependency: Path to a wheel file or a pip dependency specifier (e.g.
        "setuptools==18.0").
      temp_dir: Path to temporary installation location (optional).

  Returns:
      Temporary directory where the package was installed, that should be added
      to the Python import path.
  """
  logging.info('Installing %r to a temporary directory.', pip_dependency)
  if not temp_dir:
    temp_dir = tempfile.mkdtemp()
  install_command = [
      sys.executable, '-m', 'pip', 'install', '--target', temp_dir,
      pip_dependency
  ]
  logging.info('Executing: %s', install_command)
  subprocess.check_call(install_command)
  logging.info('Successfully installed %r.', pip_dependency)
  return temp_dir
