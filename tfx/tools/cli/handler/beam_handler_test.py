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
"""Tests for tfx.tools.cli.handler.beam_handler."""

import json
import os
import subprocess
import sys
from unittest import mock

import tensorflow as tf
from tfx.dsl.components.base import base_driver
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import beam_handler
from tfx.utils import test_case_utils


class BeamHandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self._home = self.tmp_dir
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self.enter_context(test_case_utils.override_env_var('HOME', self._home))
    self._beam_home = os.path.join(os.environ['HOME'], 'beam')
    self.enter_context(
        test_case_utils.override_env_var('BEAM_HOME', self._beam_home))

    # Flags for handler.
    self.engine = 'beam'
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_beam_1.py')
    self.pipeline_name = 'chicago_taxi_beam'
    self.pipeline_root = os.path.join(self._home, 'tfx', 'pipelines',
                                      self.pipeline_name)
    self.run_id = 'dummyID'

    self.pipeline_args = {
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
    }

  def testSavePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler._save_pipeline({labels.PIPELINE_NAME: self.pipeline_name})
    self.assertTrue(
        fileio.exists(
            os.path.join(handler._handler_home_dir, self.pipeline_name)))

  def testCreatePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()
    self.assertTrue(
        fileio.exists(handler._get_pipeline_args_path(self.pipeline_name)))

  def testCreatePipelineExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" already exists.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  def testUpdatePipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1
    }
    handler = beam_handler.BeamHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_beam_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2
    }
    handler = beam_handler.BeamHandler(flags_dict_2)
    handler.update_pipeline()
    self.assertTrue(
        fileio.exists(handler._get_pipeline_args_path(self.pipeline_name)))

  def testUpdatePipelineNoPipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  def testCompilePipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())

  def testCompilePipelineNoPipelineArgs(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: 'wrong_pipeline_path.py'
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaisesRegex(SystemExit, 'Invalid pipeline path'):
      handler.compile_pipeline()

  def testDeletePipeline(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    # Now delete the pipeline created aand check if pipeline folder is deleted.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.delete_pipeline()
    self.assertFalse(
        fileio.exists(handler._get_pipeline_info_path(self.pipeline_name)))

  def testDeletePipelineNonExistentPipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  def testListPipelinesNonEmpty(self):
    # First create two pipelines in the dags folder.
    handler_pipeline_path_1 = os.path.join(os.environ['BEAM_HOME'],
                                           'pipeline_1')
    handler_pipeline_path_2 = os.path.join(os.environ['BEAM_HOME'],
                                           'pipeline_2')
    fileio.makedirs(handler_pipeline_path_1)
    fileio.makedirs(handler_pipeline_path_2)

    # Now, list the pipelines
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = beam_handler.BeamHandler(flags_dict)

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('pipeline_1', captured.contents())
    self.assertIn('pipeline_2', captured.contents())

  def testListPipelinesEmpty(self):
    flags_dict = {labels.ENGINE_FLAG: self.engine}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_pipelines()
    self.assertIn('No pipelines to display.', captured.contents())

  def testPipelineSchemaNoPipelineRoot(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.'
    )

  def testPipelineSchemaNoSchemaGenOutput(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    fileio.makedirs(self.pipeline_root)
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Either SchemaGen component does not exist or pipeline is still running. If pipeline is running, then wait for it to successfully finish.'
    )

  def testPipelineSchemaSuccessfulRun(self):
    # First create a pipeline.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path
    }
    handler = beam_handler.BeamHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
    }
    handler = beam_handler.BeamHandler(flags_dict)
    # Create fake schema in pipeline root.
    component_output_dir = os.path.join(self.pipeline_root, 'SchemaGen')
    schema_path = base_driver._generate_output_uri(  # pylint: disable=protected-access
        component_output_dir, 'schema', 3)

    fileio.makedirs(schema_path)
    with open(os.path.join(schema_path, 'schema.pbtxt'), 'w') as f:
      f.write('SCHEMA')
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_schema()
      curr_dir_path = os.path.abspath('schema.pbtxt')
      self.assertIn('Path to schema: {}'.format(curr_dir_path),
                    captured.contents())
      self.assertIn(
          '*********SCHEMA FOR {}**********'.format(self.pipeline_name.upper()),
          captured.contents())
      self.assertTrue(fileio.exists(curr_dir_path))

  @mock.patch.object(subprocess, 'call', autospec=True, return_value=0)
  def testCreateRun(self, mock_call):
    # Create a pipeline in dags folder.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    fileio.makedirs(handler_pipeline_path)

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with open(handler._get_pipeline_args_path(self.pipeline_name), 'w') as f:
      json.dump(self.pipeline_args, f)

    # Now run the pipeline
    handler.create_run()

    mock_call.assert_called_once()
    self.assertIn(self.pipeline_path, mock_call.call_args[0][0])

  def testCreateRunNoPipeline(self):
    # Run pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name
    }
    handler = beam_handler.BeamHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.create_run()
    self.assertEqual(
        str(err.exception), 'Pipeline "{}" does not exist.'.format(
            flags_dict[labels.PIPELINE_NAME]))

  def testDeleteRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    fileio.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.delete_run()
    self.assertIn('Not supported for beam orchestrator.', captured.contents())

  def testTerminateRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    fileio.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.terminate_run()
    self.assertIn('Not supported for beam orchestrator.', captured.contents())

  def testListRuns(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    fileio.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_runs()
    self.assertIn('Not supported for beam orchestrator.', captured.contents())

  def testGetRun(self):
    # Create a pipeline in beam home.
    handler_pipeline_path = os.path.join(
        os.environ['BEAM_HOME'], self.pipeline_args[labels.PIPELINE_NAME])
    fileio.makedirs(handler_pipeline_path)

    # Now run the pipeline
    flags_dict = {labels.ENGINE_FLAG: self.engine, labels.RUN_ID: self.run_id}
    handler = beam_handler.BeamHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_run()
    self.assertIn('Not supported for beam orchestrator.', captured.contents())


if __name__ == '__main__':
  tf.test.main()
