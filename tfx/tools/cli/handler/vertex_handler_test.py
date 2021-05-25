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

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import vertex_handler
from tfx.utils import test_case_utils

_TEST_PIPELINE_NAME = 'chicago-taxi-kubeflow'
_TEST_PIPELINE_JOB_NAME = 'chicago_taxi_vertex_20200101000000'
_TEST_REGION = 'us-central1'
_TEST_PROJECT_1 = 'gcp_project_1'
_TEST_PROJECT_2 = 'gcp_project_2'  # _TEST_PROJECT_2 is assumed to have no runs.
_TEST_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:latest'
_DUMMY_APIKEY = 'dummy-api-key'
_TEST_JOB_FULL_NAME = 'projects/{}/locations/{}/pipelineJobs/{}'.format(
    _TEST_PROJECT_1, _TEST_REGION, _TEST_PIPELINE_JOB_NAME)

# Expected job detail page link associated with _TEST_JOB_FULL_NAME.
_VALID_LINK = 'https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/chicago_taxi_vertex_20200101000000?project=gcp_project_1'

_ILLEGALLY_NAMED_RUN = 'ThisIsNotAValidName'


class VertexHandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super(VertexHandlerTest, self).setUp()
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

    # Setting up Mock for API client, so that this Python test is hermatic.
    # subprocess Mock will be setup per-test.
    self.addCleanup(mock.patch.stopall)

  def testGetJobId(self):
    self.assertEqual(_TEST_PIPELINE_JOB_NAME,
                     vertex_handler._get_job_id(_TEST_JOB_FULL_NAME))

  def testGetJobIdInvalidName(self):
    with self.assertRaisesRegex(RuntimeError, 'Invalid job name is received.'):
      vertex_handler._get_job_id(_ILLEGALLY_NAMED_RUN)

  def testGetJobLink(self):
    self.assertEqual(
        _VALID_LINK,
        vertex_handler._get_job_link(_TEST_PROJECT_1, _TEST_REGION,
                                     _TEST_PIPELINE_JOB_NAME))

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


if __name__ == '__main__':
  tf.test.main()
