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
"""Package Setup script for slack custom component."""

from __future__ import print_function

from setuptools import find_packages
from setuptools import setup


def _make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return [
      'slackclient>=2.0.0,<2.0.1',
      'tfx>=0.23.0,<=0.24.0.dev',
      'websocket-client>=0.56,<0.60',
  ]


# Get version from version module.
with open('slack_component/version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setup(
    name='tfx_slack_component',
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
    install_requires=_make_required_install_packages(),
    python_requires='>=3.6,<4',
    packages=find_packages(),
    include_package_data=True,
    description='Customized TFX component for Slack integration',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='tfx slack',
    requires=[])
