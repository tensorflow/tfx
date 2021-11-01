# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Utilities for Python dependency and package management."""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List

import absl


from tfx import dependencies
from tfx import version
from tfx.dsl.io import fileio
from tfx.utils import io_utils


def make_beam_dependency_flags(beam_pipeline_args: List[str]) -> List[str]:
  """Make beam arguments for TFX python dependencies, if latter was not set.

  When TFX executors are used with non-local beam runners (Dataflow, Flink, etc)
  the remote runner needs to have access to TFX executors.
  This function acts as a helper to provide TFX source package to Beam if user
  does not provide that through Beam pipeline args.

  Args:
    beam_pipeline_args: original Beam pipeline args.

  Returns:
    updated Beam pipeline args with TFX dependencies added.
  """
  # TODO(b/176857256): Change guidance message once "ml-pipelines-sdk" extra
  # package specifiers are available.
  try:
    import apache_beam as beam  # pylint: disable=g-import-not-at-top
  except ModuleNotFoundError as e:
    raise Exception(
        'Apache Beam must be installed to use this functionality.') from e
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      flags=beam_pipeline_args)
  all_options = pipeline_options.get_all_options()
  for flag_name in [
      'extra_packages',
      'setup_file',
      'requirements_file',
      'worker_harness_container_image',
      'sdk_container_image',
  ]:
    if all_options.get(flag_name):
      absl.logging.info('Nonempty beam arg %s already includes dependency',
                        flag_name)
      return beam_pipeline_args
  absl.logging.info('Attempting to infer TFX Python dependency for beam')
  dependency_flags = []
  sdist_file = build_ephemeral_package()
  absl.logging.info('Added --extra_package=%s to beam args', sdist_file)
  dependency_flags.append('--extra_package=%s' % sdist_file)
  return beam_pipeline_args + dependency_flags


_ephemeral_setup_file = """
import setuptools

if __name__ == '__main__':
  setuptools.setup(
      name='tfx_ephemeral',
      version='{version}',
      packages=setuptools.find_namespace_packages(),
      install_requires=[{install_requires}],
      )
"""


def build_ephemeral_package() -> str:
  """Repackage current installation of TFX into a tfx_ephemeral sdist.

  Returns:
    Path to ephemeral sdist package.
  Raises:
    RuntimeError: if dist directory has zero or multiple files.
  """
  tmp_dir = os.path.join(tempfile.mkdtemp(), 'build', 'tfx')
  # Find the last directory named 'tfx' in this file's path and package it.
  path_split = __file__.split(os.path.sep)
  last_index = -1
  for i in range(len(path_split)):
    if path_split[i] == 'tfx':
      last_index = i
  if last_index < 0:
    raise RuntimeError('Cannot locate directory \'tfx\' in the path %s' %
                       __file__)
  tfx_root_dir = os.path.sep.join(path_split[0:last_index + 1])
  absl.logging.info('Copying all content from install dir %s to temp dir %s',
                    tfx_root_dir, tmp_dir)
  shutil.copytree(tfx_root_dir, os.path.join(tmp_dir, 'tfx'))
  # Source directory default permission is 0555 but we need to be able to create
  # new setup.py file.
  os.chmod(tmp_dir, 0o720)
  setup_file = os.path.join(tmp_dir, 'setup.py')
  absl.logging.info('Generating a temp setup file at %s', setup_file)
  install_requires = dependencies.make_required_install_packages()
  io_utils.write_string_file(
      setup_file,
      _ephemeral_setup_file.format(
          version=version.__version__, install_requires=install_requires))

  # Create the package
  curdir = os.getcwd()
  os.chdir(tmp_dir)
  temp_log = os.path.join(tmp_dir, 'setup.log')
  with open(temp_log, 'w') as f:
    absl.logging.info('Creating temporary sdist package, logs available at %s',
                      temp_log)
    cmd = [sys.executable, setup_file, 'sdist']
    subprocess.call(cmd, stdout=f, stderr=f)
  os.chdir(curdir)

  # Return the package dir+filename
  dist_dir = os.path.join(tmp_dir, 'dist')
  files = fileio.listdir(dist_dir)
  if not files:
    raise RuntimeError('Found no package files in %s' % dist_dir)
  elif len(files) > 1:
    raise RuntimeError('Found multiple package files in %s' % dist_dir)

  return os.path.join(dist_dir, files[0])
