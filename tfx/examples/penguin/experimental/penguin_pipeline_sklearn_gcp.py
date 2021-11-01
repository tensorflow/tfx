# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Penguin example using TFX on GCP."""

import os
from typing import Dict, List, Optional

import absl
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

# Identifier for the pipeline. This will also be used as the model name on AI
# Platform, so it should begin with a letter and only consist of letters,
# numbers, and underscores.
_pipeline_name = 'penguin_sklearn_gcp'

# Google Cloud Platform project id to use when deploying this pipeline. Leave
# blank to run locally.
_project_id = 'PROJECT_ID'

# Directory and data locations (uses Google Cloud Storage).
_bucket = 'gs://BUCKET'

# Custom container image in Google Container Registry (GCR) to use for training
# on Google Cloud AI Platform.
_tfx_image = f'gcr.io/{_project_id}/tfx-example-sklearn'

# Region to use for Dataflow jobs and AI Platform jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    # Override the default TFX image used for training with one with the correct
    # scikit-learn version.
    'masterConfig': {
        'imageUri': _tfx_image,
    },
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': _pipeline_name,
    'project_id': _project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [_gcp_region],
    # TODO(b/176256164): Update to runtime version 2.4 once that is available
    # to align with the version of TF supported by TFX.
    # LINT.IfChange
    'runtime_version': '2.3',
    # LINT.ThenChange(../../../dependencies.py)
}

# This example assumes that Penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(_bucket, 'penguin')
_data_root = os.path.join(_penguin_root, 'data')

# Python module file to inject customized logic into the TFX components.
# Trainer requires user-defined functions to run successfully.
_trainer_module_file = os.path.join(
    _penguin_root, 'experimental', 'penguin_utils_sklearn.py')

# Python module file to inject customized logic into the TFX components. The
# Evaluator component needs a custom extractor in order to make predictions
# using the scikit-learn model.
_evaluator_module_file = os.path.join(
    _penguin_root, 'experimental', 'sklearn_predict_extractor.py')

# Directory and data locations. This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem. The AI Platform Pusher requires
# that pipeline outputs are stored in a GCS bucket.
_tfx_root = os.path.join(_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

# Pipeline arguments for Beam powered Components.
# TODO(b/171316320): Change direct_running_mode back to multi_processing and set
# direct_num_workers to 0. Additionally, try to use the Dataflow runner instead
# of the direct runner.
_beam_pipeline_args = [
    '--direct_running_mode=multi_threading',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=1',
]


def _create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    trainer_module_file: str,
    evaluator_module_file: str,
    ai_platform_training_args: Optional[Dict[str, str]],
    ai_platform_serving_args: Optional[Dict[str, str]],
    beam_pipeline_args: List[str],
) -> tfx.dsl.Pipeline:
  """Implements the Penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(
      input_base=os.path.join(data_root, 'labelled'))

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # TODO(humichael): Handle applying transformation component in Milestone 3.

  # Uses user-provided Python function that trains a model.
  # Num_steps is not provided during evaluation because the scikit-learn model
  # loads and evaluates the entire test set at once.
  trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
      module_file=trainer_module_file,
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=2000),
      eval_args=tfx.proto.EvalArgs(),
      custom_config={
          tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
          ai_platform_training_args,
      })

  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

  # Uses TFMA to compute evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='species')],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='Accuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.6}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])

  evaluator = tfx.components.Evaluator(
      module_file=evaluator_module_file,
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.experimental
          .PUSHER_SERVING_ARGS_KEY: ai_platform_serving_args,
      })

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          trainer,
          model_resolver,
          evaluator,
          pusher,
      ],
      enable_cache=True,
      beam_pipeline_args=beam_pipeline_args,
  )


# To run this pipeline from the python CLI:
# $ tfx pipeline create \
#   --engine kubeflow \
#   --pipeline-path penguin_pipeline_sklearn_gcp.py \
#   --endpoint my-gcp-endpoint.pipelines.googleusercontent.com
# See TFX CLI guide for creating TFX pipelines:
# https://github.com/tensorflow/tfx/blob/master/docs/guide/cli.md#create
# For endpoint, see guide on connecting to hosted AI Platform Pipelines:
# https://cloud.google.com/ai-platform/pipelines/docs/connecting-with-sdk
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
      tfx_image=_tfx_image)

  tfx.orchestration.experimental.KubeflowDagRunner(config=runner_config).run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          trainer_module_file=_trainer_module_file,
          evaluator_module_file=_evaluator_module_file,
          ai_platform_training_args=_ai_platform_training_args,
          ai_platform_serving_args=_ai_platform_serving_args,
          beam_pipeline_args=_beam_pipeline_args))
