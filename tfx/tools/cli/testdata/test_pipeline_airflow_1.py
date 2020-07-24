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
"""Pipeline for testing CLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

# TODO(b/158143615): importing airflow after kerastuner causes issue.
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner  # pylint: disable=g-bad-import-order

from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.tools.cli.e2e import test_utils


_pipeline_name = 'chicago_taxi_simple'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data/simple')

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines')
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')
_log_root = os.path.join(_tfx_root, 'logs')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}


def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  return pipeline.Pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      components=test_utils.create_e2e_components(_data_root),
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          _metadata_path),
  )


# Airflow checks 'DAG' keyword for finding the dag.
airflow_pipeline = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline())
