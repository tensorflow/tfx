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
"""Definition for Airflow Pipeline for TFX."""
import collections
import os

from airflow import models
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.utils import logging_utils


# TODO(b/126566908): More documentation for Airflow modules.
def _get_default_metadata_connection_config(metadata_db_root, pipeline_name):
  db_uri = os.path.join(metadata_db_root, pipeline_name, 'metadata.db')
  tf.gfile.MakeDirs(os.path.dirname(db_uri))
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.filename_uri = db_uri
  connection_config.sqlite.connection_mode = \
    metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
  return connection_config


class AirflowPipeline(models.DAG):
  """TFX Pipeline for airflow.

  This is a prototype of the TFX DSL syntax onto an airflow implementation.
  """

  def __init__(self,
               pipeline_name,
               start_date,
               schedule_interval,
               pipeline_root,
               metadata_db_root,
               metadata_connection_config=None,
               additional_pipeline_args=None,
               docker_operator_cfg=None,
               enable_cache=False):
    super(AirflowPipeline, self).__init__(
        dag_id=pipeline_name,
        schedule_interval=schedule_interval,
        start_date=start_date)
    self.project_path = os.path.join(pipeline_root, pipeline_name)
    self.additional_pipeline_args = additional_pipeline_args
    self.docker_operator_cfg = docker_operator_cfg
    self.enable_cache = enable_cache

    if additional_pipeline_args is None:
      additional_pipeline_args = {}

    # Configure logging
    self.logger_config = logging_utils.LoggerConfig(pipeline_name=pipeline_name)
    if 'logger_args' in additional_pipeline_args:
      self.logger_config.update(additional_pipeline_args.get('logger_args'))

    self._logger = logging_utils.get_logger(self.logger_config)
    self.metadata_connection_config = metadata_connection_config or _get_default_metadata_connection_config(
        metadata_db_root, pipeline_name)
    self._producer_map = {}
    self._consumer_map = {}
    self._upstreams_map = collections.defaultdict(set)

  def add_node_to_graph(self, node, consumes, produces):
    """Build the dependency graph as nodes are defined."""

    consumers = self._consumer_map
    producers = self._producer_map

    # Because the entire output list is consumed as a whole,
    for artifact_list in consumes or []:
      for artifact in artifact_list:
        # register worker as a consumer of artifact(s)
        if artifact in producers:
          for other_node in producers[artifact]:
            if artifact not in consumers or node not in consumers[artifact]:
              # we should add other_node -> node
              if other_node in self._upstreams_map[node]:
                continue
              self._upstreams_map[node].add(other_node)
              node.set_upstream(other_node)

        if artifact in consumers:
          consumers[artifact].add(node)
        else:
          consumers[artifact] = {node}

    for produce_list in produces or []:
      for artifact in produce_list:
        if artifact in producers:
          producers[artifact].add(node)
        else:
          producers[artifact] = {node}
