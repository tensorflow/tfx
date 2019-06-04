# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.orchestration.kubeflow.executor_wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import tempfile
from unittest.mock import patch
import tensorflow as tf
from typing import Text

from tfx.components.example_gen.csv_example_gen import executor
from tfx.orchestration.kubeflow import executor_wrappers
from tfx.utils import types


def _locate_setup_file_dir() -> Text:
  """Traverse path from current file until setup.py is located."""
  path = os.path.dirname(__file__)
  while path != os.path.dirname(path):
    setup_file = os.path.join(path, 'setup.py')
    if os.path.isfile(setup_file):
      return path
    path = os.path.dirname(path)
  raise IOError('setup.py file not found among current file path %s' % __file__)


class ExecutorWrappersTest(tf.test.TestCase):

  def setUp(self):
    self.exec_properties = {
        'beam_pipeline_args': ['--some_flag=foo'],
        'input': json.dumps({}),
        'output': json.dumps({}),
        'output_dir': '/path/to/output',
    }
    self.examples = [types.TfxArtifact(type_name='ExamplesPath', split='dummy')]
    self.output_basedir = tempfile.mkdtemp()

    os.environ['WORKFLOW_ID'] = 'mock_workflow_id'
    os.environ['TFX_SRC_DIR'] = _locate_setup_file_dir()

  def testCsvExampleGenWrapper(self):
    input_base = types.TfxArtifact(type_name='ExternalPath', split='')
    input_base.uri = '/path/to/dataset'

    with patch.object(executor, 'Executor', autospec=True) as _:
      wrapper = executor_wrappers.CsvExampleGenWrapper(
          argparse.Namespace(
              exec_properties=json.dumps(self.exec_properties),
              outputs=types.jsonify_tfx_type_dict({'examples': self.examples}),
              executor_class_path=(
                  'tfx.components.example_gen.csv_example_gen.executor.Executor'
              ),
              input_base=json.dumps([input_base.json_dict()])
          ),
      )
      wrapper.run(output_basedir=self.output_basedir)

      # TODO(b/133011207): Validate arguments for executor and Do() method.

      metadata_file = os.path.join(
          self.output_basedir, 'output/ml_metadata/examples')

      expected_output_examples = types.TfxArtifact(
          type_name='ExamplesPath', split='dummy')
      # Expect that span and path are resolved.
      expected_output_examples.span = 1
      expected_output_examples.uri = (
          '/path/to/output/csv_example_gen/examples/mock_workflow_id/dummy/')

      with tf.gfile.GFile(metadata_file) as f:
        self.assertEqual(
            [expected_output_examples.json_dict()], json.loads(f.read()))

  # TODO(b/133011207): Test cases for other wrapper classes.


if __name__ == '__main__':
  tf.test.main()
