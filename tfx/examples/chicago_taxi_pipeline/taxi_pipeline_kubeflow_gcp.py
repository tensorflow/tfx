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
"""Chicago Taxi example using TFX DSL on Kubeflow with Google Cloud services."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
from typing import Dict, List, Optional, Text
from absl import app
from absl import flags
import tensorflow_model_analysis as tfma

from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.orchestration import data_types
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

FLAGS = flags.FLAGS
flags.DEFINE_bool('distributed_training', False,
                  'If True, enable distributed training.')

_pipeline_name = 'chicago_taxi_pipeline_kubeflow_gcp'

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_tfx_root = os.path.join(_output_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'my-gcp-project'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
_module_file = os.path.join(_input_bucket, 'taxi_utils.py')

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
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use a
    # a public container image matching the installed version of TFX.
    # 'masterConfig': { 'imageUri': 'gcr.io/my-project/my-container' },
    # Note that if you do specify a custom container, ensure the entrypoint
    # calls into TFX's run_executor script (tfx/scripts/run_executor.py)
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'chicago_taxi',
    'project_id': _project_id,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    # Note that serving currently only supports a single region:
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.models#Model
    'regions': [_gcp_region],
}


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    module_file: Text,
    ai_platform_training_args: Dict[Text, Text],
    ai_platform_serving_args: Dict[Text, Text],
    beam_pipeline_args: Optional[List[Text]] = None) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline. Should be a valid GCS path.
    module_file: uri of the module files used in Trainer and Transform
      components.
    ai_platform_training_args: Args of CAIP training job. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      for detailed description.
    ai_platform_serving_args: Args of CAIP model deployment. Please refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
      for detailed description.
    beam_pipeline_args: Optional list of beam pipeline options. Please refer to
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options.
      When this argument is not provided, the default is to use GCP
      DataflowRunner with 50GB disk size as specified in this function. If an
      empty list is passed in, default specified by Beam will be used, which can
      be found at
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#setting-other-cloud-dataflow-pipeline-options

  Returns:
    A TFX pipeline object.
  """

  # The rate at which to sample rows from the Taxi dataset using BigQuery.
  # The full taxi dataset is > 200M record.  In the interest of resource
  # savings and time, we've set the default for this example to be much smaller.
  # Feel free to crank it up and process the full dataset!
  # By default it generates a 0.1% random sample.
  query_sample_rate = data_types.RuntimeParameter(
      name='query_sample_rate', ptype=float, default=0.001)

  # This is the upper bound of FARM_FINGERPRINT in Bigquery (ie the max value of
  # signed int64).
  max_int64 = '0x7FFFFFFFFFFFFFFF'

  # The query that extracts the examples from BigQuery. The Chicago Taxi dataset
  # used for this example is a public dataset available on Google AI Platform.
  # https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips
  query = """
          SELECT
            pickup_community_area,
            fare,
            EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
            EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
            EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
            UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            trip_miles,
            pickup_census_tract,
            dropoff_census_tract,
            payment_type,
            company,
            trip_seconds,
            dropoff_community_area,
            tips
          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
          WHERE (ABS(FARM_FINGERPRINT(unique_key)) / {max_int64})
            < {query_sample_rate}""".format(
                max_int64=max_int64, query_sample_rate=str(query_sample_rate))

  # Beam args to run data processing on DataflowRunner.
  #
  # TODO(b/151114974): Remove `disk_size_gb` flag after default is increased.
  # TODO(b/151116587): Remove `shuffle_mode` flag after default is changed.
  # TODO(b/156874687): Remove `machine_type` after IP addresses are no longer a
  #                    scaling bottleneck.
  if beam_pipeline_args is None:
    beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--project=' + _project_id,
        '--temp_location=' + os.path.join(_output_bucket, 'tmp'),
        '--region=' + _gcp_region,

        # Temporary overrides of defaults.
        '--disk_size_gb=50',
        '--experiments=shuffle_mode=auto',
        '--machine_type=n1-standard-8',
    ]

  # Number of epochs in training.
  train_steps = data_types.RuntimeParameter(
      name='train_steps',
      default=10000,
      ptype=int,
  )

  # Number of epochs in evaluation.
  eval_steps = data_types.RuntimeParameter(
      name='eval_steps',
      default=5000,
      ptype=int,
  )

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = big_query_example_gen_component.BigQueryExampleGen(query=query)

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

  # Update ai_platform_training_args if distributed training was enabled.
  # Number of worker machines used in distributed training.
  worker_count = data_types.RuntimeParameter(
      name='worker_count',
      default=2,
      ptype=int,
  )

  # Type of worker machines used in distributed training.
  worker_type = data_types.RuntimeParameter(
      name='worker_type',
      default='standard',
      ptype=str,
  )

  local_training_args = copy.deepcopy(ai_platform_training_args)

  if FLAGS.distributed_training:
    local_training_args.update({
        # You can specify the machine types, the number of replicas for workers
        # and parameter servers.
        # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#ScaleTier
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'workerType': worker_type,
        'parameterServerType': 'standard',
        'workerCount': worker_count,
        'parameterServerCount': 1
    })

  # Uses user-provided Python function that implements a model using TF-Learn
  # to train a model on Google Cloud AI Platform.
  trainer = Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_trainer_executor.Executor),
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args={'num_steps': train_steps},
      eval_args={'num_steps': eval_steps},
      custom_config={
          ai_platform_trainer_executor.TRAINING_ARGS_KEY:
              local_training_args
      })

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

  # Checks whether the model passed the validation steps and pushes the model
  # to  Google Cloud AI Platform if check passed.
  # TODO(b/162451308): Add pusher back to components list once AIP Prediction
  # Service supports TF>=2.3.
  _ = Pusher(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_pusher_executor.Executor),
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      custom_config={
          ai_platform_pusher_executor.SERVING_ARGS_KEY: ai_platform_serving_args
      })

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, transform,
          trainer, model_resolver, evaluator
      ],
      beam_pipeline_args=beam_pipeline_args,
  )


def main(unused_argv):
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
      tfx_image=tfx_image)

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
      create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          module_file=_module_file,
          ai_platform_training_args=_ai_platform_training_args,
          ai_platform_serving_args=_ai_platform_serving_args,
      ))


if __name__ == '__main__':
  app.run(main)
