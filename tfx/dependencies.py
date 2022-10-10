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
      'absl-py>=0.9,<2.0.0',
      'ml-metadata' + select_constraint(
          # LINT.IfChange
          default='>=1.10.0,<1.11.0',
          # LINT.ThenChange(tfx/workspace.bzl)
          nightly='>=1.11.0.dev',
          git_master='@git+https://github.com/google/ml-metadata@master'),
      'packaging>=20,<21',
      'portpicker>=1.3.1,<2',
      'protobuf>=3.13,<4',
      'docker>=4.1,<5',
      'google-apitools>=0.5,<1',
      'google-api-python-client>=1.8,<2',
      # TODO(b/176812386): Deprecate usage of jinja2 for placeholders.
      'jinja2>=2.7.3,<4',
      # typing-extensions allows consistent & future-proof interface for typing.
      # Since kfp<2 uses typing-extensions<4, lower bound is the latest 3.x, and
      # upper bound is <5 as the semver started from 4.0 according to their doc.
      'typing-extensions>=3.10.0.2,<5',
  ]


def make_required_install_packages():
  # Make sure to sync the versions of common dependencies (absl-py, numpy,
  # and protobuf) with TF.
  return make_pipeline_sdk_required_install_packages() + [
      'apache-beam[gcp]>=2.40,<3',
      'attrs>=19.3.0,<22',
      'click>=7,<8',
      # TODO(b/245393802): Remove pinned version when pip can find depenencies
      # without this. `google-api-core` is needed for many google cloud
      # packages. `google-api-core==1.33.0` and
      # `google-cloud-aiplatform==1.18.0` requires
      # `protobuf>=3.20.1` while `tensorflow` requires `protobuf<3.20`.
      'google-api-core<1.33',
      'google-cloud-aiplatform>=1.6.2,<1.18',
      'google-cloud-bigquery>=2.26.0,<3',
      'grpcio>=1.28.1,<2',
      'keras-tuner>=1.0.4,<2',
      'kubernetes>=10.0.1,<13',
      'numpy>=1.16,<2',
      'pyarrow>=6,<7',
      'pyyaml>=3.12,<6',
      # Keep the TF version same as TFT to help Pip version resolution.
      # Pip might stuck in a TF 1.15 dependency although there is a working
      # dependency set with TF 2.x without the sync.
      # pylint: disable=line-too-long
      'tensorflow' + select_constraint(
          '>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<2.11'),
      # pylint: enable=line-too-long
      'tensorflow-hub>=0.9.0,<0.13',
      'tensorflow-data-validation' + select_constraint(
          default='>=1.10.0,<1.11.0',
          nightly='>=1.11.0.dev',
          git_master='@git+https://github.com/tensorflow/data-validation@master'
      ),
      'tensorflow-model-analysis' + select_constraint(
          default='>=0.41.0,<0.42.0',
          nightly='>=0.42.0.dev',
          git_master='@git+https://github.com/tensorflow/model-analysis@master'),
      'tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,<3',
      'tensorflow-transform' + select_constraint(
          default='>=1.10.1,<1.11.0',
          nightly='>=1.11.0.dev',
          git_master='@git+https://github.com/tensorflow/transform@master'),
      'tfx-bsl' + select_constraint(
          default='>=1.10.1,<1.11.0',
          nightly='>=1.11.0.dev',
          git_master='@git+https://github.com/tensorflow/tfx-bsl@master'),
  ]


def make_extra_packages_airflow():
  """Prepare extra packages needed for Apache Airflow orchestrator."""
  return [
      'apache-airflow[mysql]>=1.10.14,<3',
  ]


def make_extra_packages_kfp():
  """Prepare extra packages needed for Kubeflow Pipelines orchestrator."""
  return [
      'kfp>=1.8.5,<2',
      'kfp-pipeline-spec>=0.1.10,<0.2',
  ]


def make_extra_packages_test():
  """Prepare extra packages needed for running unit tests."""
  # Note: It is okay to pin packages to exact versions in this list to minimize
  # conflicts.
  return make_extra_packages_airflow() + make_extra_packages_kfp() + [
      'pytest>=5,<7',
  ]


def make_extra_packages_docker_image():
  # Packages needed for tfx docker image.
  return [
      'kfp-pipeline-spec>=0.1.10,<0.2',
      'mmh>=2.2,<3',
      'python-snappy>=0.5,<0.6',
      # Required for tfx/examples/penguin/penguin_utils_cloud_tuner.py
      'tensorflow-cloud>=0.1,<0.2',
      'tensorflow-io>=0.9.0, <=0.24.0',
  ]


def make_extra_packages_tfjs():
  # Packages needed for tfjs.
  return [
      'tensorflowjs>=3.6.0,<4',
  ]


def make_extra_packages_tflite_support():
  # Required for tfx/examples/cifar10
  return [
      'flatbuffers>=1.12,<3',
      'tflite-support>=0.4.2,<0.4.3',
  ]


def make_extra_packages_tf_ranking():
  # Packages needed for tf-ranking which is used in tfx/examples/ranking.
  return [
      'tensorflow-ranking>=0.5,<0.6',
      'struct2tensor' + select_constraint(
          default='>=0.41,<0.42',
          nightly='>=0.42.0.dev',
          git_master='@git+https://github.com/google/struct2tensor@master'),
  ]


def make_extra_packages_tfdf():
  # Packages needed for tensorflow-decision-forests.
  # Required for tfx/examples/penguin/penguin_utils_tfdf_experimental.py
  return [
      # NOTE: TFDF 1.0.1 is only compatible with TF 2.10.x.
      'tensorflow-decision-forests==1.0.1',
  ]


def make_extra_packages_flax():
  # Packages needed for the flax example.
  # Required for the experimental tfx/examples using Flax, e.g.,
  # tfx/examples/penguin.
  return [
      'jax<1',
      'jaxlib<1',
      'flax<1',
      'optax<1',
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
      # Required for tfx/examples/penguin/experimental
      # LINT.IfChange
      'scikit-learn>=0.23,<0.24',
      # LINT.ThenChange(
      #     examples/penguin/experimental/penguin_pipeline_sklearn_gcp.py)
      # Required for tfx/examples/penguin/penguin_utils_cloud_tuner.py
      'tensorflow-cloud>=0.1,<0.2',
  ]


def make_extra_packages_all():
  # All extra dependencies.
  return [
      *make_extra_packages_test(),
      *make_extra_packages_tfjs(),
      *make_extra_packages_tflite_support(),
      *make_extra_packages_tf_ranking(),
      *make_extra_packages_tfdf(),
      *make_extra_packages_flax(),
      *make_extra_packages_examples(),
  ]
