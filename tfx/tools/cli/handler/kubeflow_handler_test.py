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
"""Tests for tfx.tools.cli.handler.kubeflow_handler."""

import datetime
import os
import sys
from unittest import mock

import kfp
import tensorflow as tf

from tfx.dsl.components.base import base_driver
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import kubeflow_dag_runner_patcher
from tfx.tools.cli.handler import kubeflow_handler
from tfx.utils import test_case_utils


class _MockRunResponse:

  def __init__(self, pipeline_name, run_id, status, created_at):
    self.pipeline_spec = mock.MagicMock()
    self.pipeline_spec.pipeline_name = pipeline_name
    self.id = run_id
    self.status = status
    self.created_at = created_at


class KubeflowHandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()

    # Flags for handler.
    self.engine = 'kubeflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')

    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self.enter_context(
        test_case_utils.override_env_var('KFP_E2E_BASE_CONTAINER_IMAGE',
                                         'dummy-image'))
    self.enter_context(
        test_case_utils.override_env_var('KFP_E2E_BUCKET_NAME', 'dummy-bucket'))
    self.enter_context(
        test_case_utils.override_env_var('KFP_E2E_TEST_DATA_ROOT',
                                         'dummy-root'))

    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_kubeflow_1.py')
    self.pipeline_name = 'chicago_taxi_pipeline_kubeflow'

    # Kubeflow client params.
    self.endpoint = 'dummyEndpoint'
    self.namespace = 'kubeflow'
    self.iap_client_id = 'dummyID'

    self.runtime_parameter = {'a': '1', 'b': '2'}

    default_flags = {
        labels.ENGINE_FLAG: self.engine,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }

    self.flags_with_name = {
        **default_flags,
        labels.PIPELINE_NAME: self.pipeline_name,
    }

    self.flags_with_runtime_param = {
        **default_flags,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.RUNTIME_PARAMETER: self.runtime_parameter,
    }

    self.flags_with_dsl_path = {
        **default_flags,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
    }

    # Pipeline args for mocking subprocess.
    self.pipeline_args = {'pipeline_name': 'chicago_taxi_pipeline_kubeflow'}
    self.pipeline_id = 'the_pipeline_id'
    self.experiment_id = 'the_experiment_id'
    self.pipeline_version_id = 'the_pipeline_version_id'

    mock_client_cls = self.enter_context(
        mock.patch.object(kfp, 'Client', autospec=True))
    self.mock_client = mock_client_cls.return_value
    # Required to access generated apis.
    self.mock_client._experiment_api = mock.MagicMock()

    self.mock_client.get_pipeline_id.return_value = self.pipeline_id
    self.mock_client.get_experiment.return_value.id = self.experiment_id
    versions = [mock.MagicMock()]
    versions[0].id = self.pipeline_version_id
    self.mock_client.list_pipeline_versions.return_value.versions = versions

  def testCreatePipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_dsl_path)

    self.mock_client.get_pipeline_id.return_value = None
    self.mock_client.upload_pipeline.return_value.id = 'new_pipeline_id'

    handler.create_pipeline()

    self.mock_client.upload_pipeline.assert_called_once_with(
        pipeline_package_path=mock.ANY,
        pipeline_name=self.pipeline_name)
    self.mock_client.create_experiment.assert_called_once_with(
        self.pipeline_name)
    self.mock_client.upload_pipeline_version.assert_not_called()

  def testCreatePipelineExistentPipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_dsl_path)

    # 'the_pipeline_id' will be returned.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertIn(
        f'Pipeline "{self.pipeline_args[labels.PIPELINE_NAME]}" already exists.',
        str(err.exception))
    self.mock_client.upload_pipeline.assert_not_called()

  def testUpdatePipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_dsl_path)

    # Update test_pipeline and run update_pipeline
    handler.update_pipeline()

    self.mock_client.upload_pipeline.assert_not_called()
    self.mock_client.create_experiment.assert_not_called()
    self.mock_client.upload_pipeline_version.assert_called_once_with(
        pipeline_package_path=mock.ANY,
        pipeline_version_name=mock.ANY,
        pipeline_id=self.pipeline_id)

  def testUpdatePipelineNoPipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_dsl_path)

    self.mock_client.get_pipeline_id.return_value = None

    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertIn(f'Cannot find pipeline "{self.pipeline_name}".',
                  str(err.exception))

    self.mock_client.upload_pipeline.assert_not_called()
    self.mock_client.upload_pipeline_version.assert_not_called()

  def testCompilePipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_dsl_path)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())
    self.assertIn('Pipeline package path', captured.contents())

  def testDeletePipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_name)

    handler.delete_pipeline()

    self.mock_client.delete_pipeline.assert_called_once_with(self.pipeline_id)
    self.mock_client._experiment_api.delete_experiment.assert_called_once_with(
        self.experiment_id)

  def testDeletePipelineNonExistentPipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_name)

    self.mock_client.get_pipeline_id.return_value = None

    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertIn(f'Cannot find pipeline "{self.pipeline_name}".',
                  str(err.exception))
    self.mock_client.delete_pipeline.assert_not_called()
    self.mock_client._experiment_api.delete_experiment.assert_not_called()

  @mock.patch.object(
      kubeflow_handler.KubeflowHandler, 'execute_dsl', autospec=True)
  def testGetSchema(self, mock_execute_dsl):
    temp_pipeline_root = os.path.join(self.tmp_dir, 'pipeline_root')

    handler = kubeflow_handler.KubeflowHandler(
        {labels.ENGINE_FLAG: self.engine})
    assert isinstance(handler, kubeflow_handler.KubeflowHandler)
    mock_execute_dsl.return_value = {
        kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher.PIPELINE_NAME:
            self.pipeline_name,
        kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher.PIPELINE_ROOT:
            temp_pipeline_root
    }

    # No pipeline root
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.'
    )

    # No SchemaGen output.
    fileio.makedirs(temp_pipeline_root)
    with self.assertRaises(SystemExit) as err:
      handler.get_schema()
    self.assertEqual(
        str(err.exception),
        'Either SchemaGen component does not exist or pipeline is still running. If pipeline is running, then wait for it to successfully finish.'
    )

    # Successful pipeline run.
    # Create fake schema in pipeline root.
    component_output_dir = os.path.join(temp_pipeline_root, 'SchemaGen')
    schema_path = base_driver._generate_output_uri(  # pylint: disable=protected-access
        component_output_dir, 'schema', 3)
    fileio.makedirs(schema_path)
    with open(os.path.join(schema_path, 'schema.pbtxt'), 'w') as f:
      f.write('SCHEMA')
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.get_schema()
      curr_dir_path = os.path.join(os.getcwd(), 'schema.pbtxt')
      self.assertIn('Path to schema: {}'.format(curr_dir_path),
                    captured.contents())
      self.assertIn(
          '*********SCHEMA FOR {}**********'.format(
              self.pipeline_name.upper()), captured.contents())
      self.assertTrue(fileio.exists(curr_dir_path))

  def testCreateRun(self):
    self.mock_client.run_pipeline.return_value = _MockRunResponse(
        self.pipeline_name, '1', 'Success', datetime.datetime.now())

    handler = kubeflow_handler.KubeflowHandler(self.flags_with_runtime_param)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.create_run()
    self.assertIn('Run created for pipeline: ', captured.contents())
    self.mock_client.run_pipeline.assert_called_once_with(
        experiment_id=self.experiment_id,
        job_name=self.pipeline_name,
        params={
            'a': '1',
            'b': '2'
        },
        version_id=self.pipeline_version_id)

  def testCreateRunNoPipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_name)

    self.mock_client.get_pipeline_id.return_value = None

    with self.assertRaises(SystemExit) as err:
      handler.create_run()
    self.assertIn(f'Cannot find pipeline "{self.pipeline_name}".',
                  str(err.exception))
    self.mock_client.run_pipeline.assert_not_called()

  def testListRuns(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_name)

    self.mock_client.list_runs.return_value.runs = [
        _MockRunResponse(self.pipeline_name, '1', 'Success',
                         datetime.datetime.now()),
        _MockRunResponse(self.pipeline_name, '2', 'Failed',
                         datetime.datetime.now()),
    ]

    with self.captureWritesToStream(sys.stdout) as captured:
      handler.list_runs()

    self.mock_client.list_runs.assert_called_once_with(
        experiment_id=self.experiment_id)
    self.assertIn('pipeline_name', captured.contents())

  def testListRunsNoPipeline(self):
    handler = kubeflow_handler.KubeflowHandler(self.flags_with_name)

    self.mock_client.get_pipeline_id.return_value = None

    with self.assertRaises(SystemExit) as err:
      handler.list_runs()
    self.assertIn(f'Cannot find pipeline "{self.pipeline_name}".',
                  str(err.exception))


if __name__ == '__main__':
  tf.test.main()
