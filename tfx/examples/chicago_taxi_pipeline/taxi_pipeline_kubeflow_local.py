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
"""Chicago Taxi example using TFX DSL on Kubeflow (runs locally on cluster)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

from kfp import onprem
import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import InfraValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import infra_validator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'chicago_taxi_pipeline_kubeflow_local'

# This sample assumes a persistent volume (PV) is mounted as follows. To use
# InfraValidator with PVC, its access mode should be ReadWriteMany.
_persistent_volume_claim = 'my-pvc'
_persistent_volume = 'my-pv'
_persistent_volume_mount = '/mnt'

# All input and output data are kept in the PV.
_input_base = os.path.join(_persistent_volume_mount, 'tfx')
_output_base = os.path.join(_persistent_volume_mount, 'pipelines')
_tfx_root = os.path.join(_output_base, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

# Training data is assumed to be in ./data/simple/*.csv in the PV.
_data_root = os.path.join(_input_base, 'data', 'simple')

# Python module file to inject customized logic into the TFX components.
# The Transform and Trainer both require user-defined functions to run
# successfully. Copy taxi_utils.py to the PV in this directory.
_module_file = os.path.join(_input_base, 'taxi_utils.py')

# Path which can be listened to by the model server.
# Pusher will output the trained model here.
_serving_model_dir = os.path.join(_output_base, _pipeline_name, 'serving_model')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     module_file: Text, serving_model_dir: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""
  examples = external_input(data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model using TF-Learn
  # to train a model on Google Cloud AI Platform.
  trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000),
  )

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
                  'accuracy':
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

  # Performs infra validation of a candidate model to prevent unservable model
  # from being pushed. In order to use InfraValidator component, persistent
  # volume and its claim that the pipeline is using should be a ReadWriteMany
  # access mode.
  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=example_gen.outputs['examples'],
      serving_spec=infra_validator_pb2.ServingSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServing(
              tags=['latest']),
          kubernetes=infra_validator_pb2.KubernetesConfig()),
      request_spec=infra_validator_pb2.RequestSpec(
          tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec())
  )

  # Checks whether the model passed the validation steps and pushes the model
  # to  Google Cloud AI Platform if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      infra_blessing=infra_validator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return pipeline.Pipeline(
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
      beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':
  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

  # This pipeline automatically injects the Kubeflow TFX image if the
  # environment variable 'KUBEFLOW_TFX_IMAGE' is defined. Currently, the tfx
  # cli tool exports the environment variable to pass to the pipelines.
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      # Specify custom docker image to use.
      tfx_image=tfx_image,
      pipeline_operator_funcs=(
          # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
          # kubeflow_dag_runner.get_default_pipeline_operator_funcs() provides
          # default configurations specifically for GKE on GCP, such as secrets.
          [
              onprem.mount_pvc(_persistent_volume_claim, _persistent_volume,
                               _persistent_volume_mount)
          ]))

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          serving_model_dir=_serving_model_dir,
          beam_pipeline_args=_beam_pipeline_args))
