# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.deployment_config_utils."""

import tensorflow as tf
from tfx.orchestration.experimental.core import deployment_config_utils
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import platform_config_pb2

from google.protobuf import message

_NODE_ID = 'test-node'


def make_deployment_config(
    node_config: message.Message, node_id: str = _NODE_ID
) -> pipeline_pb2.IntermediateDeploymentConfig:
  result = pipeline_pb2.IntermediateDeploymentConfig()
  result.node_level_platform_configs[node_id].Pack(node_config)
  return result


class DeploymentConfigUtilsTest(tf.test.TestCase):

  def test_returns_none_pipeline_platform_config(self):
    self.assertIsNone(
        deployment_config_utils.get_pipeline_platform_config(
            pipeline_pb2.IntermediateDeploymentConfig()
        )
    )

  def test_returns_plain_platform_config(self):
    expected_config = platform_config_pb2.DockerPlatformConfig(
        docker_server_url='docker/server/url'
    )
    self.assertEqual(
        expected_config,
        deployment_config_utils.get_node_platform_config(
            make_deployment_config(expected_config), _NODE_ID
        ),
    )

  def test_returns_none_when_missing_platform_config(self):
    self.assertIsNone(
        deployment_config_utils.get_node_platform_config(
            pipeline_pb2.IntermediateDeploymentConfig(), _NODE_ID
        )
    )

  def test_returns_plain_executor_spec(self):
    expected_spec = executable_spec_pb2.ContainerExecutableSpec(
        image='test-docker-image'
    )
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    deployment_config.executor_specs[_NODE_ID].Pack(expected_spec)
    self.assertEqual(
        expected_spec,
        deployment_config_utils.get_node_executor_spec(
            deployment_config, _NODE_ID
        ),
    )

  def test_returns_none_when_missing_executor_spec(self):
    self.assertIsNone(
        deployment_config_utils.get_node_executor_spec(
            pipeline_pb2.IntermediateDeploymentConfig(), _NODE_ID
        )
    )


if __name__ == '__main__':
  tf.test.main()
