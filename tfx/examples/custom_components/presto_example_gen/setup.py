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
"""Package Setup script for presto based example gen component."""

from __future__ import print_function

from distutils import spawn
import os
import subprocess
import sys

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
  protoc = spawn.find_executable('protoc')


def generate_proto():
  """Invokes the Protocol Compiler to generate a _pb2.py."""
  source = 'proto/presto_config.proto'

  if protoc is None:
    sys.stderr.write(
        'protoc is not installed nor found in ../src.  Please compile it '
        'or install the binary package.\n')
    sys.exit(-1)

  protoc_command = [protoc, '-I.', '--python_out=.', source]
  if subprocess.call(protoc_command) != 0:
    sys.exit(-1)


generate_proto()

# Get version from version module.
with open('presto_component/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tfx_presto_example_gen',
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
    install_requires=[
        'presto-python-client>=0.7,<0.8',
        'tfx>=0.23.0,<=0.24.0.dev',
    ],
    python_requires='>=3.5,<4',
    packages=find_packages(),
    include_package_data=True,
    description='Customized TFX component for data ingestion from Presto',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tfx presto',
    requires=[])
