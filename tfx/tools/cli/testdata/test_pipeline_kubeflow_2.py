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
"""Chicago Taxi example using TFX DSL on Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner
from tfx.utils.dsl_utils import csv_input

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data/simple')

_output_dir = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_output_dir, 'tfx')

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'my-gcp-project'

# Region to use for Dataflow jobs and AI Platform training jobs.
_gcp_region = 'us-central1'


def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX and Kubeflow Pipelines."""

  examples = csv_input(_data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs['output'])

  return pipeline.Pipeline(
      pipeline_name='chicago_taxi_pipeline_kubeflow',
      pipeline_root=_pipeline_root,
      components=[example_gen, statistics_gen, infer_schema],
      additional_pipeline_args={
          'beam_pipeline_args': [
              '--runner=DataflowRunner',
              '--experiments=shuffle_mode=auto',
              '--project=' + _project_id,
              '--temp_location=' + os.path.join(_output_dir, 'tmp'),
              '--region=' + _gcp_region,
          ],
      },
      log_root='/var/tmp/tfx/logs',
  )


_ = KubeflowDagRunner().run(_create_pipeline())
