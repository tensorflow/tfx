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
"""Utilities to handle configurations to KubeflowDagRunner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.kubeflow.proto import kubeflow_pb2


def _get_config_value(config_value: kubeflow_pb2.ConfigValue) -> Text:
  value_from = config_value.WhichOneof('value_from')

  if value_from is None:
    raise ValueError('No value set in config value: {}'.format(config_value))

  if value_from == 'value':
    return config_value.value

  return os.getenv(config_value.environment_variable)


def get_metadata_connection_config(
    kubeflow_metadata_config: kubeflow_pb2.KubeflowMetadataConfig
) -> metadata_store_pb2.ConnectionConfig:
  """Constructs a metadata connection config.

  Args:
    kubeflow_metadata_config: Configuration parameters to use for constructing a
      valid metadata connection config in a Kubeflow cluster.

  Returns:
    A metadata_store_pb2.ConnectionConfig object.
  """
  connection_config = metadata_store_pb2.ConnectionConfig()

  connection_config.mysql.host = _get_config_value(
      kubeflow_metadata_config.mysql_db_service_host)
  connection_config.mysql.port = int(
      _get_config_value(kubeflow_metadata_config.mysql_db_service_port))
  connection_config.mysql.database = _get_config_value(
      kubeflow_metadata_config.mysql_db_name)
  connection_config.mysql.user = _get_config_value(
      kubeflow_metadata_config.mysql_db_user)
  connection_config.mysql.password = _get_config_value(
      kubeflow_metadata_config.mysql_db_password)

  return connection_config
