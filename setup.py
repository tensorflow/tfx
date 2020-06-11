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

from setuptools import find_packages
from setuptools import setup

from tfx import dependencies
from tfx import version


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
    setup_requires=['pytest-runner'],
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

