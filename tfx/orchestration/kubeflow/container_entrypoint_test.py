# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.kubeflow import container_entrypoint
from tfx.orchestration.kubeflow.proto import kubeflow_pb2


class MLMDConfigTest(tf.test.TestCase):

  def _set_required_env_vars(self, env_vars):
    for k, v in env_vars.items():
      os.environ[k] = v

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


if __name__ == '__main__':
  tf.test.main()
