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
"""Tests for tfx.orchestration.kubeflow.config_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.kubeflow import config_utils
from tfx.orchestration.kubeflow.proto import kubeflow_pb2


class ConfigUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(ConfigUtilsTest, self).setUp()
    os.environ['MYSQL_SERVICE_HOST'] = 'mysqlhost'
    os.environ['MYSQL_SERVICE_PORT'] = '3306'

  def testGetMetadataConnectionConfig(self):
    kfmd_config = kubeflow_pb2.KubeflowMetadataConfig()
    kfmd_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    kfmd_config.mysql_db_service_port.environment_variable = 'MYSQL_SERVICE_PORT'
    kfmd_config.mysql_db_name.value = 'mysqldb'
    kfmd_config.mysql_db_user.value = 'mlmduser'
    kfmd_config.mysql_db_password.value = 'mlmdpasswd'

    expected_connection_config = metadata_store_pb2.ConnectionConfig(
        mysql=metadata_store_pb2.MySQLDatabaseConfig(
            host='mysqlhost',
            port=3306,
            database='mysqldb',
            user='mlmduser',
            password='mlmdpasswd'))

    self.assertEqual(
        expected_connection_config.SerializeToString(),
        config_utils.get_metadata_connection_config(
            kfmd_config).SerializeToString())


if __name__ == '__main__':
  tf.test.main()
