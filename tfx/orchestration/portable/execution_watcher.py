# Copyright 2021 Google LLC. All Rights Reserved.
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
"""This module provides a gRPC service for updating remote job info to MLMD."""

from concurrent import futures

from typing import Optional
from absl import logging
import grpc
from tfx.proto.orchestration import execution_watcher_pb2
from tfx.proto.orchestration import execution_watcher_pb2_grpc


class ExecutionWatcher(
    execution_watcher_pb2_grpc.ExecutionWatcherServiceServicer):
  """A gRPC service server for updating remote job info to MLMD.

  Attributes:
    local_address: Local network address to the server.
    address: Remote network address to the server, same as local_address if not
             configured.
    stopped: Whether the server has been stopped or not. Stopped servers can not
             be restarted.
  """

  def __init__(self,
               port: int,
               address: Optional[str] = None,
               creds: Optional[grpc.ServerCredentials] = None):
    """Initializes the gRPC server.

    Args:
      port: Which port the service will be using.
      address: Remote address used to contact the server. Should be formatted as
               an ipv4 or ipv6 address in the format `address:port`. If left as
               None, server will use local address.
      creds: gRPC server credentials. If left as None, server will use an
             insecure port.
    """
    super().__init__()
    self._port = port
    self._address = address
    self._creds = creds
    self._server = self._create_server()
    self.stopped = False

  def UpdateExecutionInfo(self, req, unused_context):
    """Call back for executor operator to update execution info."""
    # TODO(ericlege): implement this rpc to log updates to MLMD.
    del unused_context
    logging.info('Received request to update execution info: updates %s, '
                 'execution_id %s', req.updates, req.execution_id)
    return execution_watcher_pb2.UpdateExecutionInfoResponse()

  def _create_server(self):
    """Creates a gRPC server and add `self` on to it."""
    result = grpc.server(futures.ThreadPoolExecutor())
    execution_watcher_pb2_grpc.add_ExecutionWatcherServiceServicer_to_server(
        self, result)
    if self._creds is None:
      result.add_insecure_port(self.local_address)
    else:
      result.add_secure_port(self.local_address, self._creds)
    return result

  @property
  def local_address(self) -> str:
    # Local network address to the server.
    return f'[::]:{self._port}'

  @property
  def address(self) -> str:
    return self._address or self.local_address

  def start(self):
    """Starts the server."""
    self._server.start()

  def stop(self):
    """Stops the server."""
    self._server.stop(grace=None)
    self.stopped = True

