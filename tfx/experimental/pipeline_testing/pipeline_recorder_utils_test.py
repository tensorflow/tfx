# Lint as: python3
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
"""Tests for tfx.experimental.pipeline_testing.pipeline_recorder_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import mock
import tensorflow as tf

from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.utils import io_utils


class PipelineRecorderUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(PipelineRecorderUtilsTest, self).setUp()
    self._base_dir = os.path.join(self.get_temp_dir(), 'base_dir')
    self.src_uri = os.path.join(self._base_dir, 'input')
    self.dest_uri = os.path.join(self._base_dir, 'output')
    tf.io.gfile.makedirs(self.src_uri)
    tf.io.gfile.makedirs(self.dest_uri)
    # Writing a string to test.txt file in src_uri
    self.content = 'pipeline recorded'
    io_utils.write_string_file(
        os.path.join(self.src_uri, 'test.txt'), self.content)
    # Placeholders for record_pipeline(...) arguments
    self.metadata_db_uri = 'metadata_db_uri'
    self.host = 'localhost'
    self.port = 1234
    self.pipeline_name = 'pipeline_name'
    self.run_id = 'run_id'
    # Return values for mocked get_paths(...)
    self.paths = [[self.src_uri, self.dest_uri]]
    # Return values for mocked get_execution_dict(...)
    self.execution_dict = {self.run_id: []}

  @mock.patch.object(pipeline_recorder_utils, '_get_latest_executions')
  def testRecordLatestKfpPipeline(self, mock_get_latest_executions):
    # Tests recording KFP pipeline outputs for the latest execution.
    with mock.patch.object(
        pipeline_recorder_utils, '_get_paths',
        return_value=self.paths) as mock_get_paths:
      pipeline_recorder_utils.record_pipeline(
          output_dir=self._base_dir,
          metadata_db_uri=None,
          host=self.host,
          port=self.port,
          pipeline_name=self.pipeline_name,
          run_id=None)
      mock_get_paths.assert_called()
      mock_get_latest_executions.assert_called()

      files = tf.io.gfile.listdir(self.dest_uri)
      self.assertLen(files, 1)
      self.assertEqual(
          io_utils.read_string_file(os.path.join(self.dest_uri, files[0])),
          self.content)

  def testRecordKfpPipelineRunId(self):
    # Tests recording KFP pipeline outputs given a run_id.
    with mock.patch.object(pipeline_recorder_utils, '_get_execution_dict',
                           return_value=self.execution_dict
                           ) as mock_get_execution_dict,\
        mock.patch.object(pipeline_recorder_utils, '_get_paths',
                          return_value=self.paths) as mock_get_paths:
      pipeline_recorder_utils.record_pipeline(
          output_dir=self._base_dir,
          metadata_db_uri=None,
          host=self.host,
          port=self.port,
          pipeline_name=None,
          run_id=self.run_id)

      mock_get_execution_dict.assert_called()
      mock_get_paths.assert_called()

      # Verifying that test.txt has been copied from src_uri to dest_uri
      files = tf.io.gfile.listdir(self.dest_uri)
      self.assertLen(files, 1)
      self.assertEqual(
          io_utils.read_string_file(os.path.join(self.dest_uri, files[0])),
          self.content)

  @mock.patch('tfx.orchestration.metadata.sqlite_metadata_connection_config')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  @mock.patch.object(pipeline_recorder_utils, '_get_latest_executions')
  def testRecordLatestBeamPipeline(self, mock_get_latest_executions,
                                   mock_metadata, mock_config):
    # Tests recording Beam pipeline outputs for the latest execution.
    with mock.patch.object(
        pipeline_recorder_utils, '_get_paths',
        return_value=self.paths) as mock_get_paths:
      pipeline_recorder_utils.record_pipeline(
          output_dir=self._base_dir,
          metadata_db_uri=self.metadata_db_uri,
          host=None,
          port=None,
          pipeline_name=self.pipeline_name,
          run_id=None)

      mock_config.assert_called_with(self.metadata_db_uri)
      mock_metadata.assert_called()
      mock_get_paths.assert_called()
      mock_get_latest_executions.assert_called()

      # Verifying that test.txt has been copied from src_uri to dest_uri
      files = tf.io.gfile.listdir(self.dest_uri)
      self.assertLen(files, 1)
      self.assertEqual(
          io_utils.read_string_file(os.path.join(self.dest_uri, files[0])),
          self.content)

  @mock.patch('tfx.orchestration.metadata.sqlite_metadata_connection_config')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def testRecordBeamPipelineRunId(self, mock_metadata, mock_config):
    # Tests recording Beam pipeline outputs given a run_id.
    with mock.patch.object(pipeline_recorder_utils, '_get_execution_dict',
                           return_value=self.execution_dict
                           ) as mock_get_execution_dict,\
        mock.patch.object(pipeline_recorder_utils, '_get_paths',
                          return_value=self.paths
                          ) as mock_get_paths:
      pipeline_recorder_utils.record_pipeline(
          output_dir=self._base_dir,
          metadata_db_uri=self.metadata_db_uri,
          host=None,
          port=None,
          pipeline_name=None,
          run_id=self.run_id)

      mock_config.assert_called_with(self.metadata_db_uri)
      mock_metadata.assert_called()
      mock_get_execution_dict.assert_called()
      mock_get_paths.assert_called()

      # Verifying that test.txt has been copied from src_uri to dest_uri
      files = tf.io.gfile.listdir(self.dest_uri)
      self.assertLen(files, 1)
      self.assertEqual(
          io_utils.read_string_file(os.path.join(self.dest_uri, files[0])),
          self.content)


if __name__ == '__main__':
  tf.test.main()
