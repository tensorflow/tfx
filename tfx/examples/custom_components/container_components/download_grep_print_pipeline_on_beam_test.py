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
"""Container-based pipeline sample."""

import os

import absl
import tensorflow as tf

from tfx.examples.custom_components.container_components import download_grep_print_pipeline
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as pipeline_module
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


def create_pipeline():
  """Creates the pipeline object."""
  absl.logging.set_verbosity(absl.logging.INFO)

  text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd/data/tinyshakespeare/input.txt'
  pattern = 'art thou'
  component_instances = download_grep_print_pipeline.create_pipeline_component_instances(
      text_url=text_url,
      pattern=pattern,
  )

  pipeline_name = 'download-grep-print-pipelin'

  tfx_root = os.path.join(os.environ['HOME'], 'tfx_root')
  pipeline_root = os.path.join(tfx_root, 'pipelines', pipeline_name)
  # Sqlite ML-metadata db path.
  metadata_path = os.path.join(tfx_root, 'metadata', pipeline_name,
                               'metadata.db')

  pipeline = pipeline_module.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=component_instances,
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
  )
  return pipeline


def run_pipeline_on_beam():
  """Runs the pipelineon Beam."""
  pipeline = create_pipeline()
  BeamDagRunner().run(pipeline)


class PipelineTest(tf.test.TestCase):

  def test_create_pipeline(self):
    pipeline = create_pipeline()
    self.assertIsNotNone(pipeline)


if __name__ == '__main__':
  tf.test.main()
