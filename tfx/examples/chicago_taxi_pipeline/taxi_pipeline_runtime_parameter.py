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
from typing import Optional, Text

import kfp
import tensorflow_model_analysis as tfma

from tfx.components.common_nodes.resolver_node import ResolverNode
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'taxi_pipeline_with_parameters'

# Path of pipeline root, should be a GCS path.
_pipeline_root = os.path.join('gs://my-bucket', 'tfx_taxi_simple',
                              kfp.dsl.RUN_ID_PLACEHOLDER)


def _create_parameterized_pipeline(
    pipeline_name: Text,
    pipeline_root: Optional[Text] = _pipeline_root,
    enable_cache: Optional[bool] = True,
    direct_num_workers: Optional[int] = 1) -> pipeline.Pipeline:
  """Creates a simple TFX pipeline with RuntimeParameter.

  Args:
    pipeline_name: The name of the pipeline.
    pipeline_root: The root of the pipeline output.
    enable_cache: Whether to enable cache in this pipeline.
    direct_num_workers: Number of workers executing the underlying beam pipeline
      in the executors.

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
  examples = external_input(data_root)
  example_gen = CsvExampleGen(input=examples)

  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # The module file used in Transform and Trainer component is paramterized by
  # transform_module_file.
  transform = Transform(
      input_data=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module_file)

  # The numbers of steps in train_args are specified as RuntimeParameter with
  # name 'train-steps' and 'eval-steps', respectively.
  trainer = Trainer(
      module_file=trainer_module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_output=transform.outputs['transform_graph'],
      train_args={'num_steps': train_steps},
      eval_args={'num_steps': eval_steps})

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))

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
                  'binary_accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.6}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)

  pusher = Pusher(
      model_export=trainer.outputs['model'],
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
      # TODO(b/142684737): The multi-processing API might change.
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers],
  )


if __name__ == '__main__':
  _enable_cache = True
  pipeline = _create_parameterized_pipeline(
      _pipeline_name, _pipeline_root, enable_cache=_enable_cache)

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
