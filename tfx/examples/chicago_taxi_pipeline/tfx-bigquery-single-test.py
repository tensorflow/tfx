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

import absl
import os
from typing import Dict, List, Text
from tfx.components import BigQueryExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2

_pipeline_name = 'chicago_taxi_beam_with_bigquery'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data', 'simple')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_taxi_root, 'taxi_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_taxi_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

_project_id = 'my-gcp-project-237019'

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
    pipeline_name: Text, pipeline_root: Text, query: Text, metadata_path: Text,
    direct_num_workers: int):
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=query)

  return pipeline.Pipeline(
      enable_cache=False,
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
      ],
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      # TODO(b/142684737): The multi-processing API might change.
      beam_pipeline_args=[
          '--temp_location=gs://my-tfx-bucket/tmp',
          '--project=%s' % _project_id,
          '--direct_num_workers=%d' % direct_num_workers],
  )

# To run this pipeline from the python CLI:
#   $python taxi_pipeline_beam.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          metadata_path=_metadata_path,
          query=_query,
          # 0 means auto-detect based on the number of CPUs available during
          # execution time.
          direct_num_workers=0))
