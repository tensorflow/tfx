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


def make_pipeline_sdk_required_install_packages():
  return [
      'absl-py>=0.9,<0.13',
      'ml-metadata' + select_constraint(
          # LINT.IfChange
          default='>=0.28,<0.29',
          # LINT.ThenChange(tfx/workspace.bzl)
          nightly='>=0.29.0.dev',
          git_master='@git+https://github.com/google/ml-metadata@master'),
      'packaging>=20,<21',
      'protobuf>=3.12.2,<4',
      'six>=1.10,<2',
      'docker>=4.1,<5',
      # TODO(b/176812386): Deprecate usage of jinja2 for placeholders.
      'jinja2>=2.7.3,<3',
  ]


def make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # six, and protobuf) with TF.
  return make_pipeline_sdk_required_install_packages() + [
      'apache-beam[gcp]>=2.28,<3',
      'attrs>=19.3.0,<21',
      'click>=7,<8',
      'google-api-python-client>=1.7.8,<2',
      'grpcio>=1.28.1,<2',
      # TODO(b/173976603): remove pinned keras-tuner upperbound when its
      # dependency expecatation with TensorFlow is sorted out.
      'keras-tuner>=1,<1.0.2',
      'kubernetes>=10.0.1,<12',
      # TODO(b/179195488): remove numpy dependency after 1.20 migration.
      # This dependency was added only to limit numpy 1.20 installation.
      'numpy>=1.16,<1.20',
      'pyarrow>=1,<3',
      'pyyaml>=3.12,<6',
      'tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3',
      'tensorflow-hub>=0.9.0,<0.10',
      # TODO(b/159488890): remove user module-only dependency.
      'tensorflow-cloud>=0.1,<0.2',
      'tensorflow-data-validation' + select_constraint(
          default='>=0.28,<0.29',
          nightly='>=0.29.0.dev',
          git_master='@git+https://github.com/tensorflow/data-validation@master'
      ),
      'tensorflow-model-analysis' + select_constraint(
          default='>=0.28,<0.29',
          nightly='>=0.29.0.dev',
          git_master='@git+https://github.com/tensorflow/model-analysis@master'
      ),
      'tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3',
      'tensorflow-transform' + select_constraint(
          default='>=0.28,<0.29',
          nightly='>=0.29.0.dev',
          git_master='@git+https://github.com/tensorflow/transform@master'),
      'tfx-bsl' + select_constraint(
          default='>=0.28.1,<0.29',
          nightly='>=0.29.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


def make_extra_packages_test():
  """Prepare extra packages needed for running unit tests."""
  # Note: It is okay to pin packages to exact verions in this list to minimize
  # conflicts.
  return [
      # TODO(b/178137745): Delete version cap in macos when SegFault resolved.
      ('apache-airflow[mysql]>=1.10.10,<2; '
       'python_version!="3.7" or platform_system!="Darwin"'),
      ('apache-airflow[mysql]>=1.10.10,<1.10.14; '
       'python_version=="3.7" and platform_system=="Darwin"'),
      # TODO(b/172014039): Delete pinned cattrs version after we upgrade to
      # apache-airflow 1.0.14 or later.(github.com/apache/airflow/issues/11965).
      'cattrs==1.0.0',
      'kfp>=1.1.0,<2',
      'kfp-pipeline-spec>=0.1.6,<0.2',
      'pytest>=5,<6',
      # TODO(b/182848576): Delete pinned sqlalchemy after apache-airflow 2.0.2
      # or later.(github.com/apache/airflow/issues/14811)
      'sqlalchemy>=1.3, <1.4',
      # TODO(b/175740170): Delete pinned werkzeug version after using the new
      # pip resolver.
      'werkzeug==0.16.1',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'kfp-pipeline-spec>=0.1.6,<0.2',
      'mmh>=2.2,<3',
      'python-snappy>=0.5,<0.6',
  ]


def make_extra_packages_tfjs():
  # Packages needed for tfjs.
  return [
      'tensorflowjs>=2.0.1.post1,<3',
  ]


def make_extra_packages_examples():
  # Extra dependencies required for tfx/examples.
  return [
      # Required for presto ExampleGen custom component in
      # tfx/examples/custom_components/presto_example_gen
      'presto-python-client>=0.7,<0.8',
      # Required for slack custom component in
      # tfx/examples/custom_components/slack
      'slackclient>=2.8.2,<3',
      'websocket-client>=0.57,<1',
      # Required for bert examples in tfx/examples/bert
      'tensorflow-text>=1.15.1,<3',
      # Required for tfx/examples/cifar10
      'flatbuffers>=1.12,<2',
      'tflite-support>=0.1.0a1,<0.1.1',
      # Required for tfx/examples/ranking
      'tensorflow-ranking>=0.3.3,<0.4',
      'struct2tensor>=0.28,<0.29',
      # Required for tfx/examples/penguin/experimental
      'scikit-learn>=0.24,<0.25',
  ]


def make_extra_packages_all():
  # All extra dependencies.
  return [
      *make_extra_packages_test(),
      *make_extra_packages_tfjs(),
      *make_extra_packages_examples(),
  ]
