# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""IMDB Sentiment Analysis example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl

from tfx.examples.imdb import imdb_pipeline_native_keras
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.config import pipeline_config

# To run this pipeline from the python CLI:
# $python imdb_pipeline_native_keras_stub.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  record_dir = os.path.join(os.environ['HOME'],
                            'tfx/tfx/experimental/pipeline_testing/',
                            'examples/imdb_testdata')
  my_stub_launcher = \
      stub_component_launcher.get_stub_launcher_class(
          test_data_dir=record_dir,
          stubbed_component_ids=['CsvExampleGen', \
                  'StatisticsGen', 'SchemaGen', \
                  'ExampleValidator', 'Transform', \
                  'Trainer', 'Evaluator', 'Pusher'],
          stubbed_component_map={})

  BeamDagRunner(config=pipeline_config.PipelineConfig(
      supported_launcher_classes=[
          my_stub_launcher,
      ],
      )).run(
          imdb_pipeline_native_keras._create_pipeline(  # pylint: disable=protected-access
              pipeline_name=imdb_pipeline_native_keras._pipeline_name,  # pylint: disable=protected-access
              pipeline_root=imdb_pipeline_native_keras._pipeline_root,  # pylint: disable=protected-access
              data_root=imdb_pipeline_native_keras._data_root,  # pylint: disable=protected-access
              module_file=imdb_pipeline_native_keras._module_file,  # pylint: disable=protected-access
              serving_model_dir=imdb_pipeline_native_keras._serving_model_dir,  # pylint: disable=protected-access
              metadata_path=imdb_pipeline_native_keras._metadata_path,  # pylint: disable=protected-access
              beam_pipeline_args=imdb_pipeline_native_keras._beam_pipeline_args  # pylint: disable=protected-access
          ))
