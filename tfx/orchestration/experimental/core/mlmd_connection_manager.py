# Copyright 2020 Google LLC. All Rights Reserved.
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

import dataclasses
import types
from typing import Optional, Type, Callable

from tfx.orchestration import metadata


@dataclasses.dataclass
class MLMDConnectionConfig:
  """Configuration of a connection to a MLMD db."""
  owner_name: str = ''
  project_name: str = ''
  service_target: str = ''
  base_dir: str = ''

  def __repr__(self) -> str:
    return (f'{self.__class__.__name__}('
            f'owner_name={self.owner_name}, '
            f'project_name={self.project_name}, '
            f'service_target={self.service_target}, '
            f'base_dir={self.base_dir})')

  def __hash__(self):
    return hash(self.__repr__())


class MLMDConnectionManager:
  """MLMDConnectionManager managers the connections to MLMD."""

  def __init__(self,
               primary_mlmd_handle: metadata.Metadata,
               primary_mlmd_handle_config: MLMDConnectionConfig,
               create_reader_mlmd_connection_fn: Optional[Callable[
                   [MLMDConnectionConfig], metadata.Metadata]] = None):
    """Constructor of MLMDConnectionManager.

    Args:
      primary_mlmd_handle: mlmd handle to the primary mlmd db.
      primary_mlmd_handle_config: Config of the primary mlmd handle.
      create_reader_mlmd_connection_fn: Callable function for create a mlmd
        connection.
    """
    self._primary_mlmd_handle = primary_mlmd_handle
    self._primary_mlmd_handle_config = primary_mlmd_handle_config
    self._reader_mlmd_handles = {}
    self._create_reader_mlmd_connection_fn = create_reader_mlmd_connection_fn

  def __enter__(self):
    self._primary_mlmd_handle.__enter__()
    return self

  def __exit__(self,
               exc_type: Optional[Type[Exception]] = None,
               exc_value: Optional[Exception] = None,
               exc_tb: Optional[types.TracebackType] = None) -> None:
    if self._primary_mlmd_handle:
      self._primary_mlmd_handle.__exit__(exc_type, exc_value, exc_tb)

    # Exit reader handles and make sure they are recreated upon reentry.
    for _, mlmd_handle in self._reader_mlmd_handles.items():
      mlmd_handle.__exit__(exc_type, exc_value, exc_tb)
    self._reader_mlmd_handles = {}

  @property
  def primary_mlmd_handle(self) -> metadata.Metadata:
    return self._primary_mlmd_handle

  def get_mlmd_handle(
      self, owner_name: str, project_name: str,
      mlmd_service_target_name: str) -> Optional[metadata.Metadata]:
    """Gets a MLMD db handle."""
    connection_config = MLMDConnectionConfig(
        owner_name, project_name, mlmd_service_target_name, base_dir='')
    if connection_config == self._primary_mlmd_handle_config:
      return self._primary_mlmd_handle
    elif connection_config in self._reader_mlmd_handles:
      return self._reader_mlmd_handles[connection_config]
    elif self._create_reader_mlmd_connection_fn:
      reader_mlmd_handle = self._create_reader_mlmd_connection_fn(
          connection_config)
      reader_mlmd_handle.__enter__()
      self._reader_mlmd_handles[connection_config] = reader_mlmd_handle
      return self._reader_mlmd_handles.get(connection_config)

    return None
