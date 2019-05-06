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

import glob
import os
import subprocess
import sys
from distutils.spawn import find_executable
from setuptools import find_packages
from setuptools import setup

# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
elif os.path.exists('../src/protoc'):
  protoc = '../src/protoc'
elif os.path.exists('../src/protoc.exe'):
  protoc = '../src/protoc.exe'
elif os.path.exists('../vsprojects/Debug/protoc.exe'):
  protoc = '../vsprojects/Debug/protoc.exe'
elif os.path.exists('../vsprojects/Release/protoc.exe'):
  protoc = '../vsprojects/Release/protoc.exe'
else:
  protoc = find_executable('protoc')


def generate_proto(source):
  """Invokes the Protocol Compiler to generate a _pb2.py."""

  output = source.replace('.proto', '_pb2.py')

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print('Generating %s...' % output)

    if not os.path.exists(source):
      sys.stderr.write('Cannot find required file: %s\n' % source)
      sys.exit(-1)

    if protoc is None:
      sys.stderr.write(
          'protoc is not installed nor found in ../src.  Please compile it '
          'or install the binary package.\n')
      sys.exit(-1)

    protoc_command = [protoc, '-I.', '--python_out=.', source]
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)


for proto_file in glob.glob('tfx/proto/*.proto'):
  generate_proto(proto_file)


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return [
      'absl-py>=0.1.6,<1',
      'apache-beam[gcp]>=2.12,<3',
      'google-api-python-client>=1.7.8,<2',
      'ml-metadata>=0.13.2,<0.14',
      'protobuf>=3.7,<4',
      'six>=1.10,<2',
      'tensorflow-data-validation>=0.13.1,<0.14',
      'tensorflow-model-analysis>=0.13.2,<0.14',
      'tensorflow-transform>=0.13,<0.14',
  ]


def _make_required_test_packages():
  """Prepare extra packages needed for 'python setup.py test'."""
  return [
      'apache-airflow>=1.10,<2',
      'docker>=3.7,<4',
      'kfp>=0.1,<=0.1.11; python_version >= "3.0"',
      'pytest>=4.4.1,<5',
      'tensorflow>=1.13,<2',
      'tzlocal>=1.5,<2.0',
  ]


def _make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'python-snappy>=0.5,<0.6',
      'tensorflow>=1.13.1,<2',
  ]


# Get version from version module.
with open('tfx/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# TODO(b/121329572): Remove the following comment after we can guarantee the
# required versions of packages through kokoro release workflow.
# Note: In order for the README to be rendered correctly, make sure to have the
# following minimum required versions of the respective packages when building
# and uploading the zip/wheel package to PyPI:
# setuptools >= 38.6.0, wheel >= 0.31.0, twine >= 1.11.0

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tfx',
    version=__version__,
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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    extras_require={
        'docker-image': _make_extra_packages_docker_image(),
    },
    setup_requires=['pytest-runner'],
    tests_require=_make_required_test_packages(),
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    packages=find_packages(),
    include_package_data=True,
    description='TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine learning platform implemented at Google',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tensorflow tfx',
    url='https://www.tensorflow.org/tfx',
    download_url='https://github.com/tensorflow/tfx/tags',
    requires=[])
