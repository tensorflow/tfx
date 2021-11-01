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
"""Tests for tfx.orchestration.launcher.docker_component_launcher."""

import os
from unittest import mock

import docker
import tensorflow as tf
from tfx.orchestration import publisher
from tfx.orchestration.launcher import test_utils
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import docker_executor_operator
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import platform_config_pb2

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

_EXECUTOR_SEPC = text_format.Parse(
    """
    image: "gcr://test"
    commands {
      operator {
        concat_op {
          expressions {
            value {
              string_value: "google/"
            }
          }
          expressions {
            operator {
              artifact_uri_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          type: INPUT_ARTIFACT
                          key: "input"
                        }
                      }
                      index: 0
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  """, executable_spec_pb2.ContainerExecutableSpec())

_PLATFORM_CONFIG = text_format.Parse(
    """
    environment: "VAR=1"
    environment: "ANOTHER_VAR=2"
    name: "TestContainer"
    privileged: True
    user {
      username: "test_user"
    }
    volumes {
      key: "/home/user1"
      value: {
        bind: "/mnt/vol2",
        mode: "rw"
      }
    }
    volumes {
      key: "/var/www"
      value: {
        bind: "/mnt/vol1",
        mode: "ro"
      }
    }
    additional_run_args {
      key: "arg1"
      value: "value1"
    }
    additional_run_args {
      key: "arg2"
      value: "value2"
    }""", platform_config_pb2.DockerPlatformConfig())


# TODO(hongyes): add e2e testing to cover docker launcher in beam/airflow.
class DockerComponentLauncherTest(tf.test.TestCase):

  def _set_up_test_execution_info(self,
                                  input_dict=None,
                                  output_dict=None,
                                  exec_properties=None):
    return data_types.ExecutionInfo(
        input_dict=input_dict or {},
        output_dict=output_dict or {},
        exec_properties=exec_properties or {},
        execution_output_uri='/testing/executor/output/',
        stateful_working_dir='/testing/stateful/dir',
        pipeline_node=pipeline_pb2.PipelineNode(
            node_info=pipeline_pb2.NodeInfo(
                type=metadata_store_pb2.ExecutionType(name='Docker_executor'))),
        pipeline_info=pipeline_pb2.PipelineInfo(id='test_pipeline_id'))

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(docker, 'from_env', autospec=True)
  def testLaunchSucceeds(self, mock_docker_client, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_run = mock_docker_client.return_value.containers.run
    mock_run.return_value.logs.return_value = []
    mock_run.return_value.wait.return_value = {'StatusCode': 0}
    context = self._create_launcher_context()

    execution_info = self._set_up_test_execution_info(
        input_dict={'input': [context['input_artifact']]})
    context['operator'].run_executor(execution_info)

    mock_run.assert_called_once()
    _, mock_kwargs = mock_run.call_args
    self.assertEqual('gcr://test', mock_kwargs['image'])
    self.assertListEqual(['google/' + context['input_artifact'].uri],
                         mock_kwargs['command'])
    self.assertEqual(['VAR=1', 'ANOTHER_VAR=2'], mock_kwargs['environment'])
    self.assertEqual('TestContainer', mock_kwargs['name'])
    self.assertEqual(True, mock_kwargs['privileged'])
    self.assertEqual('test_user', mock_kwargs['user'])
    self.assertEqual(
        {
            '/home/user1': {
                'bind': '/mnt/vol2',
                'mode': 'rw'
            },
            '/var/www': {
                'bind': '/mnt/vol1',
                'mode': 'ro'
            }
        }, mock_kwargs['volumes'])
    self.assertEqual('value1', mock_kwargs['arg1'])
    self.assertEqual('value2', mock_kwargs['arg2'])

  @mock.patch.object(publisher, 'Publisher', autospec=True)
  @mock.patch.object(docker, 'from_env', autospec=True)
  def testLaunchWithErrorCode(self, mock_docker_client, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}
    mock_run = mock_docker_client.return_value.containers.run
    mock_run.return_value.logs.return_value = []
    mock_run.return_value.wait.return_value = {'StatusCode': 1}
    context = self._create_launcher_context()

    with self.assertRaises(RuntimeError):
      execution_info = self._set_up_test_execution_info(
          input_dict={'input': [context['input_artifact']]})
      context['operator'].run_executor(execution_info)

  def _create_launcher_context(self, component_config=None):
    test_dir = self.get_temp_dir()

    input_artifact = test_utils._InputArtifact()
    input_artifact.uri = os.path.join(test_dir, 'input')

    operator = docker_executor_operator.DockerExecutorOperator(
        _EXECUTOR_SEPC, _PLATFORM_CONFIG)

    return {'operator': operator, 'input_artifact': input_artifact}


if __name__ == '__main__':
  tf.test.main()
