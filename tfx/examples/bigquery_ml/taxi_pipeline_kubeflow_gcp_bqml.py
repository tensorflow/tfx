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

import os
from typing import Dict, List
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.extensions.google_cloud_big_query.pusher import executor as bigquery_pusher_executor
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2

_pipeline_name = 'chicago_taxi_pipeline_kubeflow_gcp'

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_tfx_root = os.path.join(_output_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'my-gcp-project'

# BigQuery dataset where the model will be deployed, needs to be created prior
# to execution.
_bq_dataset_id = 'my-bigquery-dataset'

# Name for the model to use in BigQuery, this name is used when quering the
# model.
_model_name = 'chicago_taxi'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
_module_file = os.path.join(_input_bucket, 'taxi_utils_bqml.py')

# Region to use for Dataflow jobs and AI Platform training jobs.
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

# A dict which contains the serving job parameters for Google BigQuery ML.
_bigquery_serving_args = {
    'bq_dataset_id': _bq_dataset_id,
    'model_name': _model_name,
    'project_id': _project_id,
}

# Beam args to run data processing on DataflowRunner.
#
# TODO(b/151114974): Remove `disk_size_gb` flag after default is increased.
# TODO(b/156874687): Remove `machine_type` after IP addresses are no longer a
#                    scaling bottleneck.
# TODO(b/171733562): Remove `use_runner_v2` once it is the default for Dataflow.
_beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--project=' + _project_id,
    '--temp_location=' + os.path.join(_output_bucket, 'tmp'),
    '--region=' + _gcp_region,

    # Temporary overrides of defaults.
    '--disk_size_gb=50',
    '--machine_type=e2-standard-8',
    '--experiments=use_runner_v2',
]

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.001  # Generate a 0.1% random sample.

# This is the upper bound of FARM_FINGERPRINT in Bigquery (ie the max value of
# signed int64).
_max_int64 = '0x7FFFFFFFFFFFFFFF'

# The query that extracts the examples from BigQuery.  The Chicago Taxi dataset
# used for this example is a public dataset available on Google AI Platform.
# https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips

# TODO(b/145772608) switch to use 'optional dense' feature instead of IFNULL
# Note: TFT's feature spec generation currently does not support parsing spec
# for "optional dense" input ie.dense tensor inputs with missing values. Any
# input with missing values is interpreted as a sparse tensor. This does not
# work for BigQuery as it only supports dense input for model serving. Here we
# fill in the missing values before TFX pipeline starts as a result. This is not
# a good practice as feature transformation should always be done in the graph
# to prevent training / serving skew.

_query = """
         SELECT
           IFNULL(pickup_community_area, 0) as pickup_community_area,
           fare,
           EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
           EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
           EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
           UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
           IFNULL(pickup_latitude, 0) as pickup_latitude,
           IFNULL(pickup_longitude, 0) as pickup_longitude,
           IFNULL(dropoff_latitude, 0) as dropoff_latitude,
           IFNULL(dropoff_longitude, 0) as dropoff_longitude,
           trip_miles,
           IFNULL(pickup_census_tract, 0) as pickup_census_tract,
           IFNULL(dropoff_census_tract, 0) as dropoff_census_tract,
           payment_type,
           IFNULL(company, 'NA') as company,
           IFNULL(trip_seconds, 0) as trip_seconds,
           IFNULL(dropoff_community_area, 0) as dropoff_community_area,
           tips
         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
         WHERE (ABS(FARM_FINGERPRINT(unique_key)) / {max_int64})
           < {query_sample_rate}""".format(
               max_int64=_max_int64, query_sample_rate=_query_sample_rate)


def _create_pipeline(
    pipeline_name: str, pipeline_root: str, query: str, module_file: str,
    beam_pipeline_args: List[str], ai_platform_training_args: Dict[str, str],
    bigquery_serving_args: Dict[str, str]) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = big_query_example_gen_component.BigQueryExampleGen(query=query)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model.
  # to train a model on Google Cloud AI Platform.
  trainer = Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          ai_platform_trainer_executor.Executor),
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000),
      custom_config={'ai_platform_training_args': ai_platform_training_args})

  # Uses TFMA to compute a evaluation statistics over features of a model.
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

  # Checks whether the model passed the validation steps and pushes the model
  # to  Google Cloud BigQuery ML if check passed.
  pusher = Pusher(
      custom_executor_spec=executor_spec.ExecutorClassSpec(
          bigquery_pusher_executor.Executor),
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      custom_config={'bigquery_serving_args': bigquery_serving_args})

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, transform,
          trainer, evaluator, model_validator, pusher
      ],
      beam_pipeline_args=beam_pipeline_args,
  )


if __name__ == '__main__':
  # Metadata config. The defaults works work with the installation of
  # KF Pipelines using Kubeflow. If installing KF Pipelines using the
  # lightweight deployment option, you may need to override the defaults.
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()

  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
  )

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          query=_query,
          module_file=_module_file,
          beam_pipeline_args=_beam_pipeline_args,
          ai_platform_training_args=_ai_platform_training_args,
          bigquery_serving_args=_bigquery_serving_args,
      ))
