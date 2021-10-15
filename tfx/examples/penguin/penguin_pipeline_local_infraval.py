# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Penguin example using TFX."""

import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

_pipeline_name = 'penguin_local_infraval'

# This example assumes that penguin data is stored in ~/penguin/data and the
# utility function is in ~/penguin. Feel free to customize as needed.
_penguin_root = os.path.join(os.environ['HOME'], 'penguin')
_data_root = os.path.join(_penguin_root, 'data')
# User provided schema of the input data.
_user_provided_schema = os.path.join(_penguin_root, 'schema', 'user_provided',
                                     'schema.pbtxt')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run
# successfully.
_module_file = os.path.join(_penguin_root, 'penguin_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_penguin_root, 'serving_model',
                                  _pipeline_name)

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, accuracy_threshold: float,
                     serving_model_dir: str, metadata_path: str,
                     user_provided_schema_path: str,
                     beam_pipeline_args: List[str],
                     make_warmup: bool) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(
      input_base=os.path.join(data_root, 'labelled'))

  # Computes statistics over data for visualization and example validation.
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Import user-provided schema.
  schema_gen = tfx.components.ImportSchemaGen(
      schema_file=user_provided_schema_path)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=2000),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

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
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default',
              label_key='species_xf',
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='SparseCategoricalAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': accuracy_threshold}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
      ])
  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  # Performs infra validation of a candidate model to prevent unservable model
  # from being pushed. This config will launch a model server of the latest
  # TensorFlow Serving image in a local docker engine.
  infra_validator = tfx.components.InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=tfx.proto.ServingSpec(
          tensorflow_serving=tfx.proto.TensorFlowServing(tags=['latest']),
          local_docker=tfx.proto.LocalDockerConfig()),
      request_spec=tfx.proto.RequestSpec(
          tensorflow_serving=tfx.proto.TensorFlowServingRequestSpec(),
          # If this flag is set, InfraValidator will produce a model with
          # warmup requests (in its outputs['blessing']).
          make_warmup=make_warmup))

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  if make_warmup:
    # If InfraValidator.request_spec.make_warmup = True, its output contains
    # a model so that Pusher can push 'infra_blessing' input instead of
    # 'model' input.
    pusher = tfx.components.Pusher(
        model_blessing=evaluator.outputs['blessing'],
        infra_blessing=infra_validator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
  else:
    # Otherwise, 'infra_blessing' does not contain a model and is used as a
    # conditional checker just like 'model_blessing' does. This is the typical
    # use case.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        infra_blessing=infra_validator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          transform,
          trainer,
          model_resolver,
          evaluator,
          infra_validator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_local_infraval.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tfx.orchestration.LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          accuracy_threshold=0.6,
          serving_model_dir=_serving_model_dir,
          metadata_path=_metadata_path,
          user_provided_schema_path=_user_provided_schema,
          beam_pipeline_args=_beam_pipeline_args,
          make_warmup=True))
