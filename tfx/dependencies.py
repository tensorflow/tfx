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
"""Package dependencies for TFX."""


def make_required_install_packages():
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


def make_required_test_packages():
  """Prepare extra packages needed for 'python setup.py test'."""
  return [
      'apache-airflow>=1.10,<2',
      'docker>=3.7,<4',
      'kfp>=0.1,<=0.1.11; python_version >= "3.0"',
      'pytest>=4.4.1,<5',
      'tensorflow>=1.13,<2',
      'tzlocal>=1.5,<2.0',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'python-snappy>=0.5,<0.6',
      'tensorflow>=1.13,<2',
  ]
