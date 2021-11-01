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
"""Tests for tfx.orchestration.kubeflow.container_entrypoint."""

import json
import os
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow import container_entrypoint
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow import kubeflow_metadata_adapter
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.portable import beam_executor_operator
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import python_driver_operator
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import test_case_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class MLMDConfigTest(test_case_utils.TfxTest):

  def _set_required_env_vars(self, env_vars):
    for k, v in env_vars.items():
      self.enter_context(test_case_utils.override_env_var(k, v))

  def testDeprecatedMysqlMetadataConnectionConfig(self):
    self._set_required_env_vars({
        'mysql_host': 'mysql',
        'mysql_port': '3306',
        'mysql_database': 'metadb',
        'mysql_user_name': 'root',
        'mysql_user_password': 'test'
    })

    metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    metadata_config.mysql_db_service_host.environment_variable = 'mysql_host'
    metadata_config.mysql_db_service_port.environment_variable = 'mysql_port'
    metadata_config.mysql_db_name.environment_variable = 'mysql_database'
    metadata_config.mysql_db_user.environment_variable = 'mysql_user_name'
    metadata_config.mysql_db_password.environment_variable = 'mysql_user_password'

    ml_metadata_config = container_entrypoint._get_metadata_connection_config(
        metadata_config)
    self.assertIsInstance(ml_metadata_config,
                          metadata_store_pb2.ConnectionConfig)
    self.assertEqual(ml_metadata_config.mysql.host, 'mysql')
    self.assertEqual(ml_metadata_config.mysql.port, 3306)
    self.assertEqual(ml_metadata_config.mysql.database, 'metadb')
    self.assertEqual(ml_metadata_config.mysql.user, 'root')
    self.assertEqual(ml_metadata_config.mysql.password, 'test')

  def testGrpcMetadataConnectionConfig(self):
    self._set_required_env_vars({
        'METADATA_GRPC_SERVICE_HOST': 'metadata-grpc',
        'METADATA_GRPC_SERVICE_PORT': '8080',
    })

    grpc_config = kubeflow_pb2.KubeflowGrpcMetadataConfig()
    grpc_config.grpc_service_host.environment_variable = 'METADATA_GRPC_SERVICE_HOST'
    grpc_config.grpc_service_port.environment_variable = 'METADATA_GRPC_SERVICE_PORT'
    metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    metadata_config.grpc_config.CopyFrom(grpc_config)

    ml_metadata_config = container_entrypoint._get_metadata_connection_config(
        metadata_config)
    self.assertIsInstance(ml_metadata_config,
                          metadata_store_pb2.MetadataStoreClientConfig)
    self.assertEqual(ml_metadata_config.host, 'metadata-grpc')
    self.assertEqual(ml_metadata_config.port, 8080)

  def testDumpUiMetadata(self):
    trainer = pipeline_pb2.PipelineNode()
    trainer.node_info.type.name = 'tfx.components.trainer.component.Trainer'
    model_run_out_spec = pipeline_pb2.OutputSpec(
        artifact_spec=pipeline_pb2.OutputSpec.ArtifactSpec(
            type=metadata_store_pb2.ArtifactType(
                name=standard_artifacts.ModelRun.TYPE_NAME)))
    trainer.outputs.outputs['model_run'].CopyFrom(model_run_out_spec)

    model_run = standard_artifacts.ModelRun()
    model_run.uri = 'model_run_uri'
    exec_info = data_types.ExecutionInfo(
        input_dict={},
        output_dict={'model_run': [model_run]},
        exec_properties={},
        execution_id='id')
    ui_metadata_path = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName, 'json')
    fileio.makedirs(os.path.dirname(ui_metadata_path))
    container_entrypoint._dump_ui_metadata(
        trainer, exec_info, ui_metadata_path)
    with open(ui_metadata_path) as f:
      ui_metadata = json.load(f)
      self.assertEqual('tensorboard', ui_metadata['outputs'][-1]['type'])
      self.assertEqual('model_run_uri', ui_metadata['outputs'][-1]['source'])

  def testOverrideRegisterExecution(self):
    # Mock all real operations of driver / executor / MLMD accesses.
    mock_targets = (  # (cls, method, return_value)
        (beam_executor_operator.BeamExecutorOperator, '__init__', None),
        (beam_executor_operator.BeamExecutorOperator, 'run_executor',
         execution_result_pb2.ExecutorOutput()),
        (python_driver_operator.PythonDriverOperator, '__init__', None),
        (python_driver_operator.PythonDriverOperator, 'run_driver',
         driver_output_pb2.DriverOutput()),
        (kubeflow_metadata_adapter.KubeflowMetadataAdapter, '__init__', None),
        (launcher.Launcher, '_publish_successful_execution', None),
        (launcher.Launcher, '_clean_up_stateless_execution_info', None),
        (launcher.Launcher, '_clean_up_stateful_execution_info', None),
        (outputs_utils, 'OutputsResolver', mock.MagicMock()),
        (execution_lib, 'get_executions_associated_with_all_contexts', []),
        (container_entrypoint, '_dump_ui_metadata', None),
    )
    for cls, method, return_value in mock_targets:
      self.enter_context(
          mock.patch.object(
              cls, method, autospec=True, return_value=return_value))

    mock_mlmd = self.enter_context(
        mock.patch.object(
            kubeflow_metadata_adapter.KubeflowMetadataAdapter,
            '__enter__',
            autospec=True)).return_value
    mock_mlmd.store.return_value.get_executions_by_id.return_value = [
        metadata_store_pb2.Execution()
    ]

    self._set_required_env_vars({
        'WORKFLOW_ID': 'workflow-id-42',
        'METADATA_GRPC_SERVICE_HOST': 'metadata-grpc',
        'METADATA_GRPC_SERVICE_PORT': '8080',
        container_entrypoint._KFP_POD_NAME_ENV_KEY: 'test_pod_name'
    })

    mock_register_execution = self.enter_context(
        mock.patch.object(
            execution_publish_utils, 'register_execution',
            autospec=True))

    test_ir_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'testdata',
        'two_step_pipeline_post_dehydrate_ir.json')
    test_ir = io_utils.read_string_file(test_ir_file)

    argv = [
        '--pipeline_root',
        'dummy',
        '--kubeflow_metadata_config',
        json_format.MessageToJson(
            kubeflow_dag_runner.get_default_kubeflow_metadata_config()),
        '--tfx_ir',
        test_ir,
        '--node_id',
        'BigQueryExampleGen',
        '--runtime_parameter',
        'pipeline-run-id=STRING:my-run-id',
    ]
    container_entrypoint.main(argv)

    mock_register_execution.assert_called_once()
    kwargs = mock_register_execution.call_args[1]
    self.assertEqual(
        kwargs['exec_properties'][
            container_entrypoint._KFP_POD_NAME_PROPERTY_KEY], 'test_pod_name')


if __name__ == '__main__':
  tf.test.main()
