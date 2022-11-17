# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Utils to handle multiple connections to MLMD dbs."""

import contextlib
import types
from typing import Optional, Type, Union, cast

from tfx.orchestration import metadata

from ml_metadata.proto import metadata_store_pb2


class MLMDConnectionManager:
  """MLMDConnectionManager manages the connections to MLMD.

  It shares the same connection (or Metadata handle) for the same MLMD database,
  which can be distinguished by the "identifier" of the connection config.

  The connection sharing begins from the time the manager instance __enter__ and
  ends on the __exit__. Manager can __enter__ multiple times; in such case the
  outermost __enter__ and __exit__ pair matters.
  """

  @classmethod
  def fake(cls):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    return cls(primary_connection_config=connection_config)

  @classmethod
  def sqlite(cls, path: str):
    return cls(
        primary_connection_config=(
            metadata.sqlite_metadata_connection_config(path)))

  def _get_identifier(self, connection_config: metadata.ConnectionConfigType):
    """Get identifier for the connection config."""
    if isinstance(connection_config, metadata_store_pb2.ConnectionConfig):
      db_config = cast(metadata_store_pb2.ConnectionConfig,
                       connection_config)
      kind = db_config.WhichOneof('config')
      if kind == 'fake_database':
        return (kind,)
      elif kind == 'mysql':
        return (kind,
                db_config.mysql.host,
                db_config.mysql.port,
                db_config.mysql.database)
      elif kind == 'sqlite':
        return (kind,
                db_config.sqlite.filename_uri,
                db_config.sqlite.connection_mode)
    if isinstance(connection_config,
                  metadata_store_pb2.MetadataStoreClientConfig):
      client_config = cast(metadata_store_pb2.MetadataStoreClientConfig,
                           connection_config)
      return ('grpc_client',
              client_config.host,
              client_config.port)
    raise NotImplementedError(
        f'Unknown connection config {connection_config}.')

  def __init__(
      self, primary_connection_config: metadata.ConnectionConfigType):
    """Constructor of MLMDConnectionManager.

    Args:
      primary_connection_config: Config of the primary mlmd handle.
    """
    self._primary_connection_config = primary_connection_config
    self._exit_stack = contextlib.ExitStack()
    self._mlmd_handle_by_config_id = {}
    self._enter_count = 0

  def __enter__(self):
    self._enter_count += 1
    return self

  def __exit__(self,
               exc_type: Optional[Type[Exception]] = None,
               exc_value: Optional[Exception] = None,
               exc_tb: Optional[types.TracebackType] = None) -> None:
    self._enter_count -= 1
    if not self._enter_count:
      self._exit_stack.close()
      self._mlmd_handle_by_config_id.clear()

  @property
  def primary_mlmd_handle(self) -> metadata.Metadata:
    return self._get_mlmd_handle(self._primary_connection_config)

  def _get_mlmd_handle(
      self, connection_config: metadata.ConnectionConfigType,
  ) -> metadata.Metadata:
    """Gets or creates a memoized MLMD handle for the connection config."""
    if not self._enter_count:
      raise RuntimeError(
          'MLMDConnectionManager is not entered yet. Please use with statement '
          'first before calling get_mlmd_handle().')
    config_id = self._get_identifier(connection_config)
    if config_id in self._mlmd_handle_by_config_id:
      return self._mlmd_handle_by_config_id[config_id]
    result = metadata.Metadata(connection_config)
    self._mlmd_handle_by_config_id[config_id] = result
    self._exit_stack.enter_context(result)
    return result

  def get_mlmd_service_handle(
      self, owner: str, name: str, server_address: str) -> metadata.Metadata:
    """Gets metadata handle for MLMD Service."""
    raise NotImplementedError('MLMD Service not supported.')


MLMDHandleType = Union[metadata.Metadata, MLMDConnectionManager]


def get_primary_handle(mlmd_handle: MLMDHandleType) -> metadata.Metadata:
  if isinstance(mlmd_handle, MLMDConnectionManager):
    return mlmd_handle.primary_mlmd_handle
  else:
    return mlmd_handle
