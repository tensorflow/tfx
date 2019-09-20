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
  # TODO(b/130767399): add flask once the frontend is exposed externally.
  return [
      'absl-py>=0.1.6,<1',
      'apache-beam[gcp]>=2.14,<3',
      'click>=7.0,<8',
      'google-api-python-client>=1.7.8,<2',
      'jinja2>=2.7.3,<3',
      'ml-metadata>=0.14,<0.15',
      'protobuf>=3.7,<4',
      'six>=1.10,<2',
      'tensorflow-data-validation>=0.14.1,<0.15',
      'tensorflow-model-analysis>=0.14,<0.15',
      'tensorflow-transform>=0.14,<0.15',
  ]


def make_required_test_packages():
  """Prepare extra packages needed for 'python setup.py test'."""
  return [
      'apache-airflow>=1.10,<2',
      'docker>=4.0.0,<5.0.0',
      # LINT.IfChange
      'kfp>=0.1.30,<0.2; python_version >= "3.0"',
      # LINT.ThenChange(
      #     testing/github/common.sh,
      #     testing/github/ubuntu/image/image.sh,
      #     testing/kubeflow/common.sh
      # )
      'pytest>=5.0.0,<6.0.0',
      'tensorflow>=1.14,<2',
      'tzlocal>=1.5,<2.0',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'python-snappy>=0.5,<0.6',
      'tensorflow>=1.14,<2',
      # TODO(b/138406006): Remove the narrower dependency for pyarrow
      # and numpy after Beam 2.15 release.
      'numpy>=1.16,<1.17',
      'pyarrow>=0.14,<0.15',
  ]
