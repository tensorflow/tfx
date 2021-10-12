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
          default='>=1.3.0,<1.4.0',
          # LINT.ThenChange(tfx/workspace.bzl)
          nightly='>=1.4.0.dev',
          git_master='@git+https://github.com/google/ml-metadata@master'),
      'packaging>=20,<21',
      'portpicker>=1.3.1,<2',
      'protobuf>=3.13,<4',
      'docker>=4.1,<5',
      'google-apitools>=0.5,<1',
      'google-api-python-client>=1.8,<2',
      # TODO(b/176812386): Deprecate usage of jinja2 for placeholders.
      'jinja2>=2.7.3,<4',
  ]


def make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # and protobuf) with TF.
  return make_pipeline_sdk_required_install_packages() + [
      'apache-beam[gcp]>=2.32,<3',
      'attrs>=19.3.0,<21',
      'click>=7,<8',
      'google-cloud-aiplatform>=1.5.0,<2',
      'google-cloud-bigquery>=2.26.0,<3',
      'grpcio>=1.28.1,<2',
      'keras-tuner>=1.0.4,<2',
      'kubernetes>=10.0.1,<13',
      # TODO(b/179195488): remove numpy dependency after 1.20 migration.
      # This dependency was added only to limit numpy 1.20 installation.
      'numpy>=1.16,<1.20',
      'pyarrow>=1,<6',
      'pyyaml>=3.12,<6',
      'tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3',
      'tensorflow-hub>=0.9.0,<0.13',
      'tensorflow-data-validation' + select_constraint(
          default='>=1.3.0,<1.4.0',
          nightly='>=1.4.0.dev',
          git_master='@git+https://github.com/tensorflow/data-validation@master'
      ),
      'tensorflow-model-analysis' + select_constraint(
          default='>=0.34.1,<0.35',
          nightly='>=0.35.0.dev',
          git_master='@git+https://github.com/tensorflow/model-analysis@master'),
      'tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3',
      'tensorflow-transform' + select_constraint(
          default='>=1.3.0,<1.4.0',
          nightly='>=1.4.0.dev',
          git_master='@git+https://github.com/tensorflow/transform@master'),
      'tfx-bsl' + select_constraint(
          default='>=1.3.0,<1.4.0',
          nightly='>=1.4.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


def make_extra_packages_airflow():
  """Prepare extra packages needed for Apache Airflow orchestrator."""
  return [
      # TODO(b/188940096): update supported version.
      'apache-airflow[mysql]>=1.10.14,<3',
      # TODO(b/182848576): Delete pinned sqlalchemy after apache-airflow 2.0.2
      # or later.(github.com/apache/airflow/issues/14811)
      'sqlalchemy>=1.3,<1.4',
  ]


def make_extra_packages_kfp():
  """Prepare extra packages needed for Kubeflow Pipelines orchestrator."""
  return [
      # kfp==1.7.2 uses undefined field when updating a pipeline; b/197906254.
      # TODO(b/200220058): Unblock upper bound after the version issue of
      # `typing-extensions` with TF 2.6 is resolved.
      'kfp>=1.6.1,!=1.7.2,<1.8.2',
      'kfp-pipeline-spec>=0.1.10,<0.2',
  ]


def make_extra_packages_test():
  """Prepare extra packages needed for running unit tests."""
  # Note: It is okay to pin packages to exact versions in this list to minimize
  # conflicts.
  return make_extra_packages_airflow() + make_extra_packages_kfp() + [
      'pytest>=5,<6',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'kfp-pipeline-spec>=0.1.10,<0.2',
      'mmh>=2.2,<3',
      'python-snappy>=0.5,<0.6',
      # Required for tfx/examples/penguin/penguin_utils_cloud_tuner.py
      'tensorflow-cloud>=0.1,<0.2',
  ]


def make_extra_packages_tfjs():
  # Packages needed for tfjs.
  return [
      'tensorflowjs>=3.6.0,<4',
  ]


def make_extra_packages_tf_ranking():
  # Packages needed for tf-ranking which is used in tfx/examples/ranking.
  return [
      'tensorflow-ranking>=0.3.3,<0.4',
      'struct2tensor' + select_constraint(
          default='>=0.34,<0.35',
          nightly='>=0.35.0.dev',
          git_master='@git+https://github.com/google/struct2tensor@master'),
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
      'flatbuffers>=1.12,<3',
      'tflite-support>=0.1.0a1,<0.2.1',
      # Required for tfx/examples/penguin/experimental
      # LINT.IfChange
      'scikit-learn>=0.23,<0.24',
      # LINT.ThenChange(
      #     examples/penguin/experimental/penguin_pipeline_sklearn_gcp.py)
      # Required for the experimental tfx/examples using Flax, e.g.,
      # tfx/examples/penguin.
      # TODO(b/193362300): Unblock the version cap after TF 2.7 becomes minimum.
      'jax>=0.2.13,<0.2.17',
      'jaxlib>=0.1.64,<0.2',
      'flax>=0.3.3,<0.4',
      # Required for tfx/examples/penguin/penguin_utils_cloud_tuner.py
      'tensorflow-cloud>=0.1,<0.2',
  ]


def make_extra_packages_all():
  # All extra dependencies.
  return [
      *make_extra_packages_test(),
      *make_extra_packages_tfjs(),
      *make_extra_packages_tf_ranking(),
      *make_extra_packages_examples(),
  ]
