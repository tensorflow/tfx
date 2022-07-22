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
from typing import Optional
from tfx.orchestration import metadata
from ml_metadata.google.services.mlmd_service.proto import mlmd_service_pb2


class MLMDConnectionManager:
  """MLMDConnectionManager managers the connections to MLMD."""

  def __init__(self, project_owner: str, project_name: str,
               mlmd_server_address: str, primary_mlmd_handle: metadata.Metadata,
               context_manager: contextlib.ExitStack):
    self._project_owner = project_owner
    self._project_name = project_name
    self._mlmd_server_address = mlmd_server_address
    self._primary_mlmd_handle: metadata.Metadata = primary_mlmd_handle
    self._reader_mlmd_handles = {}
    self._context_manager = context_manager

  def get_primary_mlmd_handle(self) -> metadata.Metadata:
    return self._primary_mlmd_handle

  def connect_reader_mlmd_db(self, project_owner: str, project_name: str):
    """connect to a mlmd db."""
    pipeline_asset = mlmd_service_pb2.PipelineAsset(
        owner=project_owner, name=project_name)
    mlmd_handle = metadata.Metadata(
        connection_config=mlmd_service_pb2.MLMDServiceClientConfig(
            server_address=self._mlmd_server_address,
            rpc_deadline_secs=1000,
            pipeline_asset=pipeline_asset,
            client_type=mlmd_service_pb2.ClientType(
                client_latency_type=mlmd_service_pb2.ClientLatencyType
                .CLIENT_LATENCY_TOLERANT,
                client_staleness_type=mlmd_service_pb2.ClientStalenessType
                .CLIENT_STALENESS_SENSITIVE)))
    self._context_manager.enter_context(mlmd_handle)
    self._reader_mlmd_handles[(project_owner, project_name)] = mlmd_handle

  def get_reader_mlmd_handle(self, project_owner: str,
                             project_name: str) -> Optional[metadata.Metadata]:
    return self._reader_mlmd_handles.get((project_owner, project_name))

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass
