# Lint as: python2, python3
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Text

import absl
import apache_beam as beam
import tensorflow as tf

from tfx import dependencies
from tfx import version
from tfx.utils import io_utils


def _get_pypi_package_version() -> Optional[Text]:
  """Returns package version if TFX is installed from PyPI, otherwise None."""
  # We treat any integral patch version as published to PyPI, since development
  # packages always end with 'dev' or 'rc'.
  if version.__version__.split('.')[-1].isdigit():
    return version.__version__
  else:
    return None


def make_beam_dependency_flags(beam_pipeline_args: List[Text]) -> List[Text]:
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
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      flags=beam_pipeline_args)
  all_options = pipeline_options.get_all_options()
  for flag_name in ['extra_package', 'setup_file', 'requirements_file']:
    if all_options.get(flag_name):
      absl.logging.info('Nonempty beam arg %s already includes dependency',
                        flag_name)
      return beam_pipeline_args
  absl.logging.info('Attempting to infer TFX Python dependency for beam')
  dependency_flags = []
  pypi_version = _get_pypi_package_version()
  # TODO(b/147438224): refactor once PortableRunner drops no-binary.
  if (pypi_version and '--runner=PortableRunner' not in beam_pipeline_args and
      '--runner=FlinkRunner' not in beam_pipeline_args and
      '--runner=SparkRunner' not in beam_pipeline_args):
    requirements_file = _build_requirements_file()
    absl.logging.info('Added --requirements_file=%s to beam args',
                      requirements_file)
    dependency_flags.append('--requirements_file=%s' % requirements_file)
  else:
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
      packages=setuptools.find_packages(),
      install_requires=[{install_requires}],
      )
"""


def _build_requirements_file() -> Text:
  """Returns a requirements.txt file which includes current TFX package."""
  result = os.path.join(tempfile.mkdtemp(), 'requirement.txt')
  absl.logging.info('Generating a temp requirements.txt file at %s', result)
  io_utils.write_string_file(result, 'tfx==%s' % version.__version__)
  return result


def build_ephemeral_package() -> Text:
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
  cmd = [sys.executable, setup_file, 'sdist']
  subprocess.call(cmd)
  os.chdir(curdir)

  # Return the package dir+filename
  dist_dir = os.path.join(tmp_dir, 'dist')
  files = tf.io.gfile.listdir(dist_dir)
  if not files:
    raise RuntimeError('Found no package files in %s' % dist_dir)
  elif len(files) > 1:
    raise RuntimeError('Found multiple package files in %s' % dist_dir)

  return os.path.join(dist_dir, files[0])
