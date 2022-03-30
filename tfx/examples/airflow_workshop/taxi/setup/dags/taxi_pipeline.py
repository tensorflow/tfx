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
# pylint: disable=line-too-long
# pylint: disable=unused-import
# pylint: disable=unused-argument
"""Chicago taxi example using TFX."""

import datetime
import os
from typing import List

from tfx.components import CsvExampleGen
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing

from tfx.components import StatisticsGen # Step 3
from tfx.components import SchemaGen # Step 3
from tfx.components import ExampleValidator # Step 3

from tfx.components import Transform # Step 4


from tfx.components import Trainer # Step 5
from tfx.proto import trainer_pb2 # Step 5
import tensorflow_model_analysis as tfma # Step 5

from tfx.components import Evaluator # Step 6
from tfx.dsl.components.common import resolver # Step 6
from tfx.dsl.experimental import latest_blessed_model_resolver # Step 6

from tfx.components import Pusher # Step 7
from tfx.proto import pusher_pb2 # Step 7


_pipeline_name = 'taxi'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'airflow')
_data_root = os.path.join(_taxi_root, 'data')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_taxi_root, 'dags', 'taxi_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(_taxi_root, 'tfx')
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

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


# TODO(b/137289334): rename this as simple after DAG visualization is done.
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples']) # Step 3

  # Generates schema based on statistics files.
  infer_schema = SchemaGen( # Step 3
       statistics=statistics_gen.outputs['statistics'], # Step 3
       infer_feature_shape=False) # Step 3

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator( # Step 3
       statistics=statistics_gen.outputs['statistics'], # Step 3
       schema=infer_schema.outputs['schema']) # Step 3

  # Performs transformations and feature engineering in training and serving.
  transform = Transform( # Step 4
       examples=example_gen.outputs['examples'], # Step 4
       schema=infer_schema.outputs['schema'], # Step 4
       module_file=module_file) # Step 4

  # Uses user-provided Python function that implements a model.
  trainer = Trainer( # Step 5
       module_file=module_file, # Step 5
       custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor), # Step 5
       examples=transform.outputs['transformed_examples'], # Step 5
       transform_graph=transform.outputs['transform_graph'], # Step 5
       schema=infer_schema.outputs['schema'], # Step 5
       train_args=trainer_pb2.TrainArgs(num_steps=10000), # Step 5
       eval_args=trainer_pb2.EvalArgs(num_steps=5000)) # Step 5

  # Get the latest blessed model for model validation.
  model_resolver = resolver.Resolver(# Step 6
       strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver, # Step 6
       model=Channel(type=Model), # Step 6
       model_blessing=Channel(type=ModelBlessing)).with_id( # Step 6
           'latest_blessed_model_resolver') # Step 6

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig( # Step 6
       model_specs=[ # Step 6
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and
        # remove the label_key.
        tfma.ModelSpec( # Step 6
            signature_name='serving_default', # Step 6
            label_key='tips', # Step 6
            preprocessing_function_names=['transform_features'], # Step 6
            ) # Step 6
        ], # Step 6
    metrics_specs=[ # Step 6
        tfma.MetricsSpec( # Step 6
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            metrics=[ # Step 6
                tfma.MetricConfig(class_name='ExampleCount'), # Step 6
                tfma.MetricConfig(class_name='BinaryAccuracy', # Step 6
                  threshold=tfma.MetricThreshold( # Step 6
                      value_threshold=tfma.GenericValueThreshold( # Step 6
                          lower_bound={'value': 0.5}), # Step 6
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold( # Step 6
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER, # Step 6
                          absolute={'value': -1e-10}))) # Step 6
            ] # Step 6
        ) # Step 6
    ], # Step 6
    slicing_specs=[ # Step 6
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(), # Step 6
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec( # Step 6
            feature_keys=['trip_start_hour']) # Step 6
    ]) # Step 6

  model_analyzer = Evaluator( # Step 6
       examples=example_gen.outputs['examples'], # Step 6
       model=trainer.outputs['model'], # Step 6
       baseline_model=model_resolver.outputs['model'], # Step 6
       eval_config=eval_config) # Step 6

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher( # Step 7
       model=trainer.outputs['model'], # Step 7
       model_blessing=model_analyzer.outputs['blessing'], # Step 7
       push_destination=pusher_pb2.PushDestination( # Step 7
           filesystem=pusher_pb2.PushDestination.Filesystem( # Step 7
               base_directory=serving_model_dir))) # Step 7

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen, # Step 3
          infer_schema, # Step 3
          validate_stats, # Step 3
          transform, # Step 4
          trainer, # Step 5
          model_resolver, # Step 6
          model_analyzer, # Step 6
          pusher, # Step 7
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# 'DAG' below need to be kept for Airflow to detect dag.
DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        beam_pipeline_args=_beam_pipeline_args))
