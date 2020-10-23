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
"""Utilies for user defined functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Optional, Text

from absl import logging

from tfx.utils import import_utils

# Key for module file.
_MODULE_FILE_KEY = 'module_file'
# Key for module path.
_MODULE_PATH_KEY = 'module_path'
# Ephemeral setup.py file name.
_EPHEMERAL_SETUP_PY_FILE_NAME = '_tfx_generated_setup.py'


# TODO(b/157155972): improve user code support.
def get_fn(exec_properties: Dict[Text, Any],
           fn_name: Text) -> Callable[..., Any]:
  """Loads and returns user-defined function."""

  has_module_file = bool(exec_properties.get(_MODULE_FILE_KEY))
  has_module_path = bool(exec_properties.get(_MODULE_PATH_KEY))
  has_fn = bool(exec_properties.get(fn_name))

  if has_module_path:
    module_path = exec_properties[_MODULE_PATH_KEY]
    if module_path in sys.modules:
      importlib.reload(sys.modules[module_path])
    print('!!!! imported', module_path, 'from',
          __import__(module_path).__file__, fn_name)
    return getattr(__import__(module_path), fn_name)
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


def _get_ephemeral_setup_py_contents(package_name, module_names):
  version_string = '0.%d' % int(time.time())
  return """import setuptools

setuptools.setup(
    name=%r,
    version=%r,
    author='TFX User',
    author_email='nobody@example.com',
    description='Auto-generated TFX user code package.',
    py_modules=%r,
    classifiers=[],
    python_requires='>=3.6',
)
""" % (version_string, package_name, module_names)


def package_user_module_file(instance_name: Text, module_path: Text):
  """Package the given user module file into a Python Wheel package."""
  module_path = os.path.abspath(module_path)
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

  setup_py_path = os.path.join(build_dir, _EPHEMERAL_SETUP_PY_FILE_NAME)
  with open(setup_py_path, 'w') as f:
    f.write(
        _get_ephemeral_setup_py_contents('tfx-user-code-%s' % instance_name,
                                         module_names))

  temp_dir = tempfile.mkdtemp()
  dist_dir = tempfile.mkdtemp()
  bdist_command = [
      sys.executable, setup_py_path, 'bdist_wheel', '--bdist-dir', temp_dir,
      '--dist-dir', dist_dir
  ]
  logging.info('Executing: %s', bdist_command)
  subprocess.check_call(bdist_command, cwd=build_dir)

  dist_files = os.listdir(dist_dir)
  if len(dist_files) != 1:
    raise Exception(
        'Unexpectedly found %d output files in wheel output directory %s.' %
        (len(dist_files), dist_dir))
  dist_file_path = os.path.join(dist_dir, dist_files[0])
  os.remove(setup_py_path)

  return dist_file_path, user_module_name


def install_to_temp_directory(pip_dependency):
  """Install the given pip dependency specifier to a temporary directory."""
  logging.info('Installing %r to a temporary directory.', pip_dependency)
  temp_dir = tempfile.mkdtemp()
  install_command = [
      sys.executable, '-m', 'pip', 'install', '--target', temp_dir,
      pip_dependency
  ]
  logging.info('Executing: %s', install_command)
  subprocess.check_call(install_command)
  logging.info('Successfully installed %r.', pip_dependency)
  return temp_dir
