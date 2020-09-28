# Lint as: python3
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
"""Package dependencies for TFX.

tfx and family libraries (such as tensorflow-model-analysis) adopts environment
variable (TFX_DEPENDENCY_SELECTOR) based dependency version selection. This
dependency will be baked in to the wheel, in other words you cannot change
dependency string once wheel is built.

- UNCONSTRAINED uses dependency without any version constraint string, which is
  useful when you manually build wheels of parent library (e.g. tfx-bsl) of
  arbitrary version, and install it without dependency constraints conflict.
- NIGHTLY uses x.(y+1).0.dev version as a lower version constraint. tfx nightly
  will transitively depend on nightly versions of other TFX family libraries,
  and this version constraint is required.
- GIT_MASTER uses github master branch URL of the dependency, which is useful
  during development, or when depending on the github master HEAD version of
  tfx. This is because tfx github master HEAD version is actually using github
  master HEAD version of parent libraries.
  Caveat: URL dependency is not upgraded with --upgrade flag, and you have to
  specify --force-reinstall flag to fetch the latest change from each master
  branch HEAD.
- For the release, we use a range of version, which is also used as a default.
"""
import os


def select_constraint(default, nightly=None, git_master=None):
  """Select dependency constraint based on TFX_DEPENDENCY_SELECTOR env var."""
  selector = os.environ.get('TFX_DEPENDENCY_SELECTOR')
  if selector == 'UNCONSTRAINED':
    return ''
  elif selector == 'NIGHTLY' and nightly is not None:
    return nightly
  elif selector == 'GIT_MASTER' and git_master is not None:
    return git_master
  else:
    return default


def make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  # TODO(b/130767399): add flask once the frontend is exposed externally.
  return [
      'absl-py>=0.9,<0.11',
      # LINT.IfChange
      'apache-beam[gcp]>=2.24,<3',
      # LINT.ThenChange(examples/chicago_taxi_pipeline/setup/setup_beam.sh)
      'attrs>=19.3.0,<20',
      'click>=7,<8',
      'docker>=4.1,<5',
      'google-api-python-client>=1.7.8,<2',
      'grpcio>=1.28.1,<2',
      'jinja2>=2.7.3,<3',
      'keras-tuner>=1,<2',
      'kubernetes>=10.0.1,<12',
      'ml-metadata' + select_constraint(
          # LINT.IfChange
          default='>=0.24,<0.25',
          # LINT.ThenChange(opensource_only/build/tfx.workspace.bzl)
          nightly='>=0.25.0.dev',
          git_master='@git+https://github.com/google/ml-metadata@master'),
      'protobuf>=3.12.2,<4',
      'pyarrow>=0.17,<0.18',
      'pyyaml>=3.12,<6',
      'six>=1.10,<2',
      'tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3',
      'tensorflow-data-validation' + select_constraint(
          default='>=0.24.1,<0.25',
          nightly='>=0.25.0.dev',
          git_master='@git+https://github.com/tensorflow/data-validation@master'),  # pylint: disable=line-too-long
      'tensorflow-model-analysis' + select_constraint(
          default='>=0.24.3,<0.25',
          nightly='>=0.25.0.dev',
          git_master='@git+https://github.com/tensorflow/model-analysis@master'),  # pylint: disable=line-too-long
      'tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,<3',
      'tensorflow-transform' + select_constraint(
          default='>=0.24.1,<0.25',
          nightly='>=0.25.0.dev',
          git_master='@git+https://github.com/tensorflow/transform@master'),
      'tfx-bsl' + select_constraint(
          default='>=0.24.1,<0.25',
          nightly='>=0.25.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


def make_required_test_packages():
  """Prepare extra packages needed for running unit tests."""
  # Note: It is okay to pin packages to exact verions in this list to minimize
  # conflicts.
  return [
      'apache-airflow[mysql]>=1.10.10,<2',
      # TODO(b/157033885): Remove pinned version of WTForms after newer version
      # of Apache Airflow.
      'WTForms==2.2.1',
      'kfp>=0.4,<0.5',
      'pytest>=5,<6',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'python-snappy>=0.5,<0.6',
  ]


def make_extra_packages_tfjs():
  # Packages needed for tfjs.
  return [
      'tensorflowjs>=2.0.1.post1,<3',
      # TODO(b/158034704): Remove prompt-toolkit pin resulted from
      # tfjs -> PyInquirer dependency chain.
      'prompt-toolkit>=2.0.10,<3',
  ]


def make_all_dependency_packages():
  # All extra dependencies.
  return make_required_test_packages() + make_extra_packages_tfjs()
