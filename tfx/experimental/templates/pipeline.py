# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.big_query_example_gen.component import BigQueryExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.experimental.templates import common
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2


def create_components(input_config: example_gen_pb2.Input,
                      preprocessing_fn: Text, trainer_fn: Text,
                      push_destination: pusher_pb2.PushDestination):
  """Implements the components in the pipeline."""

  # Developer TODO: Adjust this with pipeline specification.

  example_gen = BigQueryExampleGen(input_config=input_config)

  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

  schema_gen = SchemaGen(
      stats=statistics_gen.outputs['output'], infer_feature_shape=False)

  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs['output'],
      schema=schema_gen.outputs['output'])

  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=schema_gen.outputs['output'],
      preprocessing_fn=preprocessing_fn)

  trainer = Trainer(
      trainer_fn=trainer_fn,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['output'],
      transform_output=transform.outputs['transform_output'],
      train_args=common.TRAIN_ARGS,
      eval_args=common.EVAL_ARGS)

  model_evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model_exports=trainer.outputs['output'])

  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['output'])

  pusher = Pusher(
      model_export=trainer.outputs['output'],
      model_blessing=model_validator.outputs['blessing'],
      push_destination=push_destination)

  return [
      example_gen, statistics_gen, schema_gen, validate_stats, transform,
      trainer, model_evaluator, model_validator, pusher
  ]


def run_beam_pipeline():
  """Runs components with BeamDagRunner."""
  beam_dag_runner.BeamDagRunner().run(
      tfx_pipeline.Pipeline(
          pipeline_name=common.PIPELINE_NAME,
          pipeline_root=common.PIPELINE_ROOT,
          components=create_components(
              input_config=common.INPUT_CONFIG,
              preprocessing_fn=common.PREPROCESSING_FN,
              trainer_fn=common.TRAINER_FN,
              push_destination=common.PUSH_DESTINATION,
          ),
          enable_cache=common.ENABLE_CACHE,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(
              common.METADATA_ROOT),
          additional_pipaline_args=common.ADDITIONAL_PIPELINE_ARGS))
