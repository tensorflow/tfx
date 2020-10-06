# Lint as: python3
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
"""Chicago taxi example using TFX."""

import os
from typing import Text

import absl
from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.examples.custom_components.hello_world.hello_component import component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

_pipeline_name = 'taxi_hello_pipeline'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(os.path.dirname(__file__), 'data', 'simple')

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(_taxi_root, 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     metadata_path: Text) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name=u'HelloWorld')

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[example_gen, hello, statistics_gen],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path))


# To run this pipeline from the python CLI:
#   $python taxi_pipeline_hello.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          metadata_path=_metadata_path))
