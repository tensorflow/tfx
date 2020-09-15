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

import os
import subprocess

import setuptools
from setuptools import find_packages
from setuptools import setup
from setuptools.command import develop
# pylint: disable=g-bad-import-order
# It is recommended to import setuptools prior to importing distutils to avoid
# using legacy behavior from distutils.
# https://setuptools.readthedocs.io/en/latest/history.html#v48-0-0
from distutils import spawn
from distutils.command import build
# pylint: enable=g-bad-import-order

from tfx import dependencies
from tfx import version
from tfx.tools import resolve_deps


class _BuildCommand(build.build):
  """Build everything that is needed to install.

  This overrides the original distutils "build" command to to run gen_proto
  command before any sub_commands.

  build command is also invoked from bdist_wheel and install command, therefore
  this implementation covers the following commands:
    - pip install . (which invokes bdist_wheel)
    - python setup.py install (which invokes install command)
    - python setup.py bdist_wheel (which invokes bdist_wheel command)
  """

  def _should_generate_proto(self):
    """Predicate method for running GenProto command or not."""
    return True

  # Add "gen_proto" command as the first sub_command of "build". Each
  # sub_command of "build" (e.g. "build_py", "build_ext", etc.) is executed
  # sequentially when running a "build" command, if the second item in the tuple
  # (predicate method) is evaluated to true.
  sub_commands = [
      ('gen_proto', _should_generate_proto),
  ] + build.build.sub_commands


class _DevelopCommand(develop.develop):
  """Developmental install.

  https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode
  Unlike normal package installation where distribution is copied to the
  site-packages folder, developmental install creates a symbolic link to the
  source code directory, so that your local code change is immediately visible
  in runtime without re-installation.

  This is a setuptools-only (i.e. not included in distutils) command that is
  also used in pip's editable install (pip install -e). Originally it only
  invokes build_py and install_lib command, but we override it to run gen_proto
  command in advance.

  This implementation covers the following commands:
    - pip install -e . (developmental install)
    - python setup.py develop (which is invoked from developmental install)
  """

  def run(self):
    self.run_command('gen_proto')
    # Run super().initialize_options. Command is an old-style class (i.e.
    # doesn't inherit object) and super() fails in python 2.
    develop.develop.run(self)


class _GenProtoCommand(setuptools.Command):
  """Generate proto stub files in python.

  Running this command will populate foo_pb2.py file next to your foo.proto
  file.
  """

  def initialize_options(self):
    pass

  def finalize_options(self):
    self._bazel_cmd = spawn.find_executable('bazel')
    if not self._bazel_cmd:
      raise RuntimeError(
          'Could not find "bazel" binary. Please visit '
          'https://docs.bazel.build/versions/master/install.html for '
          'installation instruction.')

  def run(self):
    subprocess.check_call(
        [self._bazel_cmd, 'run', '//build:gen_proto'],
        # Bazel should be invoked in a directory containing bazel WORKSPACE
        # file, which is the root directory.
        cwd=os.path.dirname(os.path.realpath(__file__)),)


# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()


setup(
    name='tfx',
    version=version.__version__,
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=dependencies.make_required_install_packages(),
    extras_require={
        # In order to use 'docker-image' or 'all', system libraries specified
        # under 'tfx/tools/docker/Dockerfile' are required
        'docker-image': dependencies.make_extra_packages_docker_image(),
        'tfjs': dependencies.make_extra_packages_tfjs(),
        'all': dependencies.make_all_dependency_packages(),
    },
    # TODO(b/158761800): Move to [build-system] requires in pyproject.toml.
    setup_requires=[
        'pytest-runner',
        'poetry==1.0.9',  # Required for ResolveDeps command.
                          # Poetry API is not officially documented and subject
                          # to change in the future. Thus fix the version.
        'clikit>=0.4.3,<0.5',  # Required for ResolveDeps command.
    ],
    cmdclass={
        'build': _BuildCommand,
        'develop': _DevelopCommand,
        'gen_proto': _GenProtoCommand,
        'resolve_deps': resolve_deps.ResolveDepsCommand,
    },
    python_requires='>=3.5,<4',
    packages=find_packages(),
    include_package_data=True,
    description='TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine learning platform implemented at Google',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow tfx',
    url='https://www.tensorflow.org/tfx',
    download_url='https://github.com/tensorflow/tfx/tags',
    requires=[],
    # Below console_scripts, each line identifies one console script. The first
    # part before the equals sign (=) which is 'tfx', is the name of the script
    # that should be generated, the second part is the import path followed by a
    # colon (:) with the Click command group. After installation, the user can
    # invoke the CLI using "tfx <command_group> <sub_command> <flags>"
    entry_points="""
        [console_scripts]
        tfx=tfx.tools.cli.cli_main:cli_group
    """)
