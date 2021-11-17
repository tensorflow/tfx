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
"""Tests for Vertex handler."""

import os
import sys
from unittest import mock

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import vertex_handler
from tfx.utils import test_case_utils

_TEST_PIPELINE_NAME = 'chicago-taxi-vertex'
_TEST_REGION = 'us-central1'
_TEST_PROJECT_1 = 'gcp_project_1'


class VertexHandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')

    self._home = self.tmp_dir
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self.enter_context(test_case_utils.override_env_var('HOME', self._home))
    self._vertex_home = os.path.join(self._home, 'vertex')
    self.enter_context(
        test_case_utils.override_env_var('VERTEX_HOME', self._vertex_home))

    # Flags for handler.
    self.engine = 'vertex'
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_kubeflow_v2_1.py')
    self.pipeline_name = _TEST_PIPELINE_NAME
    self.pipeline_root = os.path.join(self._home, 'tfx', 'pipelines',
                                      self.pipeline_name)
    self.run_id = 'dummyID'
    self.project = 'gcp_project_1'
    self.region = 'us-central1'

    self.runtime_parameter = {'a': '1', 'b': '2'}

    # Setting up Mock for API client, so that this Python test is hermetic.
    # subprocess Mock will be setup per-test.
    self.addCleanup(mock.patch.stopall)

  def testCreatePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    handler.create_pipeline()
    self.assertTrue(
        fileio.exists(
            handler._get_pipeline_definition_path(self.pipeline_name)))

  def testCreatePipelineExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception),
        'Pipeline "{}" already exists.'.format(self.pipeline_name))

  def testUpdatePipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_v2_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1
    }
    handler = vertex_handler.VertexHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_v2_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2
    }
    handler = vertex_handler.VertexHandler(flags_dict_2)
    handler.update_pipeline()
    self.assertTrue(
        fileio.exists(
            handler._get_pipeline_definition_path(self.pipeline_name)))

  def testUpdatePipelineNoPipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception),
        'Pipeline "{}" does not exist.'.format(self.pipeline_name))

  def testCompilePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn(f'Pipeline {self.pipeline_name} compiled successfully',
                  captured.contents())

  def testListPipelinesNonEmpty(self):
    # First create two pipelines in the dags folder.
    handler_pipeline_path_1 = os.path.join(os.environ['VERTEX_HOME'],
                                           'pipeline_1')
    handler_pipeline_path_2 = os.path.join(os.environ['VERTEX_HOME'],
                                           'pipeline_2')
    fileio.makedirs(handler_pipeline_path_1)
    fileio.makedirs(handler_pipeline_path_2)

    # Now, list the pipelines
    flags_dict = {labels.ENGINE_FLAG: labels.VERTEX_ENGINE}
    handler = vertex_handler.VertexHandler(flags_dict)

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('pipeline_1', captured.contents())
    self.assertIn('pipeline_2', captured.contents())

  def testListPipelinesEmpty(self):
    flags_dict = {labels.ENGINE_FLAG: labels.VERTEX_ENGINE}
    handler = vertex_handler.VertexHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('No pipelines to display.', captured.contents())

  def testDeletePipeline(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    handler.create_pipeline()

    # Now delete the pipeline created aand check if pipeline folder is deleted.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = os.path.join(handler._handler_home_dir,
                                         self.pipeline_name)
    self.assertFalse(fileio.exists(handler_pipeline_path))

  def testDeletePipelineNonExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = vertex_handler.VertexHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  @mock.patch.object(aiplatform, 'init', autospec=True)
  @mock.patch.object(pipeline_jobs, 'PipelineJob', autospec=True)
  def testCreateRun(self, mock_pipeline_job, mock_init):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.GCP_PROJECT_ID: _TEST_PROJECT_1,
        labels.GCP_REGION: _TEST_REGION,
        labels.RUNTIME_PARAMETER: self.runtime_parameter,
    }
    # TODO(b/198114641): Delete following override after upgrading source code
    # to aiplatform>=1.3.
    mock_pipeline_job.return_value.wait_for_resource_creation = mock.MagicMock()

    handler = vertex_handler.VertexHandler(flags_dict)
    handler.create_run()

    mock_init.assert_called_once_with(
        project=_TEST_PROJECT_1, location=_TEST_REGION)
    mock_pipeline_job.assert_called_once_with(
        display_name=_TEST_PIPELINE_NAME,
        template_path=handler._get_pipeline_definition_path(
            _TEST_PIPELINE_NAME),
        parameter_values={
            'a': '1',
            'b': '2'
        })


if __name__ == '__main__':
  tf.test.main()
