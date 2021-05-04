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
"""Chicago Taxi example demonstrating the usage of RuntimeParameter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import kfp
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

_pipeline_name = 'taxi_pipeline_with_parameters'

# Path of pipeline root, should be a GCS path.
_pipeline_root = os.path.join('gs://my-bucket', 'tfx_taxi_simple',
                              kfp.dsl.RUN_ID_PLACEHOLDER)

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_parameterized_pipeline(
    pipeline_name: Text, pipeline_root: Text, enable_cache: bool,
    beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Creates a simple TFX pipeline with RuntimeParameter.

  Args:
    pipeline_name: The name of the pipeline.
    pipeline_root: The root of the pipeline output.
    enable_cache: Whether to enable cache in this pipeline.
    beam_pipeline_args: Pipeline arguments for Beam powered Components.

  Returns:
    A logical TFX pipeline.Pipeline object.
  """
  # First, define the pipeline parameters.
  # Path to the CSV data file, under which there should be a data.csv file.
  data_root = data_types.RuntimeParameter(
      name='data-root',
      default='gs://my-bucket/data',
      ptype=Text,
  )

  # Path to the transform module file.
  transform_module_file = data_types.RuntimeParameter(
      name='transform-module',
      default='gs://my-bucket/modules/transform_module.py',
      ptype=Text,
  )

  # Path to the trainer module file.
  trainer_module_file = data_types.RuntimeParameter(
      name='trainer-module',
      default='gs://my-bucket/modules/trainer_module.py',
      ptype=Text,
  )

  # Number of epochs in training.
  train_steps = data_types.RuntimeParameter(
      name='train-steps',
      default=10,
      ptype=int,
  )

  # Number of epochs in evaluation.
  eval_steps = data_types.RuntimeParameter(
      name='eval-steps',
      default=5,
      ptype=int,
  )

  # The input data location is parameterized by data_root
  example_gen = CsvExampleGen(input_base=data_root)

  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # The module file used in Transform and Trainer component is paramterized by
  # transform_module_file.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module_file)

  # The numbers of steps in train_args are specified as RuntimeParameter with
  # name 'train-steps' and 'eval-steps', respectively.
  trainer = Trainer(
      module_file=trainer_module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(Executor),
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args={'num_steps': train_steps},
      eval_args={'num_steps': eval_steps})

  # Get the latest blessed model for model validation.
  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id('latest_blessed_model_resolver')

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.6}),
                          # Change threshold will be ignored if there is no
                          # baseline model resolved from MLMD (first run).
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(
                  str(pipeline.ROOT_PARAMETER), 'model_serving'))))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, transform,
          trainer, model_resolver, evaluator, pusher
      ],
      enable_cache=enable_cache,
      beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
  pipeline = _create_parameterized_pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      enable_cache=True,
      beam_pipeline_args=_beam_pipeline_args)

  # This pipeline automatically injects the Kubeflow TFX image if the
  # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
  # cli tool exports the environment variable to pass to the pipelines.
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=kubeflow_dag_runner
      .get_default_kubeflow_metadata_config(),
      tfx_image=tfx_image)
  kfp_runner = kubeflow_dag_runner.KubeflowDagRunner(config=config)

  kfp_runner.run(pipeline)
