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

import logging
import os
import shutil
import subprocess
import sys

import setuptools
from setuptools import find_namespace_packages
from setuptools import setup
from setuptools.command import develop
# pylint: disable=g-bad-import-order
# It is recommended to import setuptools prior to importing distutils to avoid
# using legacy behavior from distutils.
# https://setuptools.readthedocs.io/en/latest/history.html#v48-0-0
from distutils.command import build
# pylint: enable=g-bad-import-order

from tfx import dependencies
from tfx import version
from wheel import bdist_wheel

# Prefer to import `package_config` from the setup.py script's directory. The
# `package_config.py` file is used to configure which package to build (see
# the logic below switching on `package_config.PACKAGE_NAME`) and the overall
# package build README at `package_build/README.md`.
sys.path.insert(0, os.path.dirname(__file__))
# pylint: disable=g-bad-import-order,g-import-not-at-top
import package_config
# pylint: enable=g-bad-import-order,g-import-not-at-top


class _BdistWheelCommand(bdist_wheel.bdist_wheel):
  """Overrided bdist_wheel command.

  Inject some custom command line arguments and flags that can be used in the
  subcommands. This command class covers:
    - pip wheel --build-option="--local-mlmd-repo=${MLMD_OUTPUT_DIR}"
    - python setup.py bdist_wheel --local-mlmd-repo="${MLMD_OUTPUT_DIR}"
  """
  user_options = bdist_wheel.bdist_wheel.user_options + [
      ('local-mlmd-repo=', None, 'Path to the local MLMD repository to use '
       'instead of the Bazel com_github_google_ml_metadata remote repository.')
  ]

  def initialize_options(self):
    # Run super().initialize_options. Command is an old-style class (i.e.
    # doesn't inherit object) and super() fails in python 2.
    bdist_wheel.bdist_wheel.initialize_options(self)
    self.local_mlmd_repo = None

  def finalize_options(self):
    bdist_wheel.bdist_wheel.finalize_options(self)
    gen_proto = self.distribution.get_command_obj('gen_proto')
    gen_proto.local_mlmd_repo = self.local_mlmd_repo


class _UnsupportedDevBuildWheelCommand(_BdistWheelCommand):
  """Disables build of 'tfx-dev' wheel files."""

  def finalize_options(self):
    if not os.environ.get('UNSUPPORTED_BUILD_TFX_DEV_WHEEL'):
      raise Exception(
          'Starting in version 0.26.0, pip package build for TFX has changed,'
          'and `python setup.py bdist_wheel` can no longer be invoked '
          'directly.\n\nFor instructions on how to build wheels for TFX, see '
          'https://github.com/tensorflow/tfx/blob/master/package_build/'
          'README.md.\n\nEditable pip installation for development is still '
          'supported through `pip install -e`.')
    super().finalize_options()


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
  user_options = [
      ('local-mlmd-repo=', None, 'Path to the local MLMD repository to use '
       'instead of the Bazel com_github_google_ml_metadata remote repository.')
  ]

  def initialize_options(self):
    self.local_mlmd_repo = None

  def finalize_options(self):
    self._bazel_cmd = shutil.which('bazel')
    if not self._bazel_cmd:
      raise RuntimeError(
          'Could not find "bazel" binary. Please visit '
          'https://docs.bazel.build/versions/master/install.html for '
          'installation instruction.')

  def run(self):
    bazel_args = ['--compilation_mode', 'opt']
    if self.local_mlmd_repo:
      # If local MLMD repo is given, override com_github_google_ml_metadata
      # remote repository with the local path. This is required to use the
      # local developmental version of MLMD during tests.
      # https://docs.bazel.build/versions/master/command-line-reference.html
      bazel_args.append('--override_repository={}={}'.format(
          'com_github_google_ml_metadata', self.local_mlmd_repo))
    cmd = [self._bazel_cmd, 'run', *bazel_args, '//build:gen_proto']
    print('Running Bazel command', cmd, file=sys.stderr)
    subprocess.check_call(
        cmd,
        # Bazel should be invoked in a directory containing bazel WORKSPACE
        # file, which is the root directory.
        cwd=os.path.dirname(os.path.realpath(__file__)),
        env=os.environ)


_TFX_DESCRIPTION = (
    'TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine '
    'learning platform implemented at Google.')
_PIPELINES_SDK_DESCRIPTION = (
    'A dependency-light distribution of the core pipeline authoring '
    'functionality of TensorFlow Extended (TFX).')

# Get the long descriptions from README files.
with open('README.md') as fp:
  _TFX_LONG_DESCRIPTION = fp.read()
with open('README.ml-pipelines-sdk.md') as fp:
  _PIPELINES_SDK_LONG_DESCRIPTION = fp.read()

package_name = package_config.PACKAGE_NAME
tfx_extras_requires = {
    # In order to use 'docker-image' or 'all', system libraries specified
    # under 'tfx/tools/docker/Dockerfile' are required
    'docker-image': dependencies.make_extra_packages_docker_image(),
    'airflow': dependencies.make_extra_packages_airflow(),
    'kfp': dependencies.make_extra_packages_kfp(),
    'tfjs': dependencies.make_extra_packages_tfjs(),
    'tf-ranking': dependencies.make_extra_packages_tf_ranking(),
    'examples': dependencies.make_extra_packages_examples(),
    'test': dependencies.make_extra_packages_test(),
    'all': dependencies.make_extra_packages_all(),
}

# Packages included the TFX namespace.
TFX_NAMESPACE_PACKAGES = [
    'tfx', 'tfx.*', 'tfx.orchestration', 'tfx.orchestration.*'
]
# Packages within the TFX namespace that are to be included in the base
# "ml-pipelines-sdk" pip package (and excluded from the "tfx" pip package,
# which takes "ml-pipelines-sdk" as a dependency).
ML_PIPELINES_SDK_PACKAGES = [
    # This adds `tfx.version` which is needed in several places.
    'tfx',
    # Core DSL subpackage.
    'tfx.dsl',
    'tfx.dsl.*',
    # The "ml-pipelines-sdk" package currently only supports local execution.
    # These are the subpackages of `tfx.orchestration` necessary.
    'tfx.orchestration',
    'tfx.orchestration.config',
    'tfx.orchestration.launcher',
    'tfx.orchestration.local',
    'tfx.orchestration.local.legacy',
    'tfx.orchestration.portable',
    'tfx.orchestration.portable.*',
    # Note that `tfx.proto` contains TFX first-party component-specific
    # protobuf definitions, but `tfx.proto.orchestration` contains portable
    # execution protobuf definitions which are needed in the base package.
    'tfx.proto.orchestration',
    # TODO(b/176814928): Consider moving relevant modules under
    # `tfx.orchestration.*` to `tfx.dsl.*` as appropriate.
    'tfx.proto.orchestration.*',
    # TODO(b/176795329): Move `tfx.utils` to a location that emphasizes that
    # these are internal utilities.
    'tfx.utils',
    'tfx.utils.*',
    # TODO(b/176795331): Move `Artifact` and `ComponentSpec` classes into
    # `tfx.dsl.*`.
    'tfx.types',
    'tfx.types.*',
]

EXCLUDED_PACKAGES = [
    'tfx.benchmarks',
    'tfx.benchmarks.*',
]

# Below console_scripts, each line identifies one console script. The first
# part before the equals sign (=) which is 'tfx', is the name of the script
# that should be generated, the second part is the import path followed by a
# colon (:) with the Click command group. After installation, the user can
# invoke the CLI using "tfx <command_group> <sub_command> <flags>"
TFX_ENTRY_POINTS = """
    [console_scripts]
    tfx=tfx.tools.cli.cli_main:cli_group
"""
ML_PIPELINES_SDK_ENTRY_POINTS = None

# This `setup.py` file can be used to build packages in 3 configurations. See
# the discussion in `package_build/README.md` for an overview. The `tfx` and
# `ml-pipelines-sdk` pip packages can be built for distribution using the
# selectable `package_config.PACKAGE_NAME` specifier. Additionally, for
# development convenience, the `tfx-dev` package containing the union of the
# the `tfx` and `ml-pipelines-sdk` package can be installed as an editable
# package using `pip install -e .`, but should not be built for distribution.
if package_config.PACKAGE_NAME == 'tfx-dev':
  # Monolithic development package with the entirety of `tfx.*` and the full
  # set of dependencies. Functionally equivalent to the union of the "tfx" and
  # "tfx-pipeline-sdk" packages.
  install_requires = dependencies.make_required_install_packages()
  extras_require = tfx_extras_requires
  description = _TFX_DESCRIPTION
  long_description = _TFX_LONG_DESCRIPTION
  packages = find_namespace_packages(
      include=TFX_NAMESPACE_PACKAGES, exclude=EXCLUDED_PACKAGES)
  # Do not support wheel builds for "tfx-dev".
  build_wheel_command = _UnsupportedDevBuildWheelCommand  # pylint: disable=invalid-name
  # Include TFX entrypoints.
  entry_points = TFX_ENTRY_POINTS
elif package_config.PACKAGE_NAME == 'ml-pipelines-sdk':
  # Core TFX pipeline authoring SDK, without dependency on component-specific
  # packages like "tensorflow" and "apache-beam".
  install_requires = dependencies.make_pipeline_sdk_required_install_packages()
  extras_require = {}
  description = _PIPELINES_SDK_DESCRIPTION
  long_description = _PIPELINES_SDK_LONG_DESCRIPTION
  packages = find_namespace_packages(
      include=ML_PIPELINES_SDK_PACKAGES, exclude=EXCLUDED_PACKAGES)
  # Use the default pip wheel building command.
  build_wheel_command = bdist_wheel.bdist_wheel  # pylint: disable=invalid-name
  # Include ML Pipelines SDK entrypoints.
  entry_points = ML_PIPELINES_SDK_ENTRY_POINTS
elif package_config.PACKAGE_NAME == 'tfx':
  # Recommended installation package for TFX. This package builds on top of
  # the "ml-pipelines-sdk" pipeline authoring SDK package and adds first-party
  # TFX components and additional functionality.
  install_requires = (['ml-pipelines-sdk==%s' % version.__version__] +
                      dependencies.make_required_install_packages())
  extras_require = tfx_extras_requires
  description = _TFX_DESCRIPTION
  long_description = _TFX_LONG_DESCRIPTION
  packages = find_namespace_packages(
      include=TFX_NAMESPACE_PACKAGES,
      exclude=ML_PIPELINES_SDK_PACKAGES + EXCLUDED_PACKAGES)
  # Use the pip wheel building command that includes proto generation.
  build_wheel_command = _BdistWheelCommand  # pylint: disable=invalid-name
  # Include TFX entrypoints.
  entry_points = TFX_ENTRY_POINTS
else:
  raise ValueError('Invalid package config: %r.' % package_config.PACKAGE_NAME)

logging.info('Executing build for package %r.', package_name)

setup(
    name=package_name,
    version=version.__version__,
    author='Google LLC',
    author_email='tensorflow-extended-dev@googlegroups.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=install_requires,
    extras_require=extras_require,
    # TODO(b/158761800): Move to [build-system] requires in pyproject.toml.
    setup_requires=[
        'pytest-runner',
    ],
    cmdclass={
        'bdist_wheel': build_wheel_command,
        'build': _BuildCommand,
        'develop': _DevelopCommand,
        'gen_proto': _GenProtoCommand,
    },
    python_requires='>=3.7,<3.9',
    packages=packages,
    include_package_data=True,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='tensorflow tfx',
    url='https://www.tensorflow.org/tfx',
    download_url='https://github.com/tensorflow/tfx/tags',
    requires=[],
    entry_points=entry_points)
