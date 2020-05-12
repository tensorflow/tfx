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
"""Package Setup script for TFX."""

from __future__ import print_function

from distutils import spawn
from distutils.command import build_py
from distutils.command import clean
import glob
import os
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup

from tfx import dependencies


def _find_proto_compiler():
  """Find protoc compiler binary path."""

  if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
    return os.environ['PROTOC']
  elif os.path.exists('../src/protoc'):
    return os.path.realpath('../src/protoc')
  elif os.path.exists('../src/protoc.exe'):
    return os.path.realpath('../src/protoc.exe')
  elif os.path.exists('../vsprojects/Debug/protoc.exe'):
    return os.path.realpath('../vsprojects/Debug/protoc.exe')
  elif os.path.exists('../vsprojects/Release/protoc.exe'):
    return os.path.realpath('../vsprojects/Release/protoc.exe')
  else:
    return spawn.find_executable('protoc')


class _ProtoCompilerMixin(object):
  """Mixins for generating py proto stubs from proto file."""
  _protoc = _find_proto_compiler()

  def generate_proto(self, source):
    """Invokes the Protocol Compiler to generate a _pb2.py.

    Args:
      source: path to the proto source file (endswith ".proto").
    """
    output = source.replace('.proto', '_pb2.py')

    if (not os.path.exists(output) or
        (os.path.exists(source) and
         os.path.getmtime(source) > os.path.getmtime(output))):
      print('Generating %s...' % output, file=sys.stderr)

      if not os.path.exists(source):
        sys.stderr.write('Cannot find required file: %s\n' % source)
        sys.exit(-1)

      if self._protoc is None:
        sys.stderr.write(
            'protoc is not installed nor found in ../src.  Please compile it '
            'or install the binary package.\n')
        sys.exit(-1)

      protoc_command = [self._protoc, '-I.', '--python_out=.', source]
      if subprocess.call(protoc_command) != 0:
        sys.exit(-1)


class _BuildPyCmd(build_py.build_py, _ProtoCompilerMixin):
  """BuildPy command with python proto stub generation."""

  _PROTO_FILE_PATTERNS = (
      'tfx/proto/*.proto',
      'tfx/orchestration/kubeflow/proto/*.proto',
  )

  def run(self):
    # Pre-command hook.
    # Generate proto python stubs (*_pb2.py).
    for file_pattern in self._PROTO_FILE_PATTERNS:
      for proto_file in glob.glob(file_pattern):
        self.generate_proto(proto_file)

    # Run build_py. It is an old-style class and super() doesn't work.
    build_py.build_py.run(self)


class _CleanCmd(clean.clean):
  """Clean command with python proto stub cleanup."""

  def run(self):
    # Pre-command hook.
    # Remove proto python stubs (*_pb2.py).
    for root, _, files in os.walk('.'):
      for name in files:
        if name.endswith('_pb2.py'):
          os.remove(os.path.join(root, name))

    # Run clean. It is an old-style class and super() doesn't work.
    clean.clean.run(self)


setup(
    namespace_packages=[],
    install_requires=dependencies.make_required_install_packages(),
    extras_require={
        # In order to use 'docker-image' or 'all', system libraries specified
        # under 'tfx/tools/docker/Dockerfile' are required
        'docker-image': dependencies.make_extra_packages_docker_image(),
        'all': dependencies.make_all_dependency_packages(),
    },
    setup_requires=['pytest-runner'],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    packages=find_packages(),
    include_package_data=True,
    requires=[],
    cmdclass={
        'clean': _CleanCmd,
        'build_py': _BuildPyCmd,
    },
    # Below console_scripts, each line identifies one console script. The first
    # part before the equals sign (=) which is 'tfx', is the name of the script
    # that should be generated, the second part is the import path followed by a
    # colon (:) with the Click command group. After installation, the user can
    # invoke the CLI using "tfx <command_group> <sub_command> <flags>"
    entry_points={
        'console_scripts': [
            'tfx = tfx.tools.cli.cli_main:cli_group'
        ],
    },
)
