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
"""Module for shared interface of every model server clients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import six
from typing import List

from tfx.components.infra_validator import types


class ModelState(enum.Enum):
  # Model is NOT_READY if it is not available at the moment, but can be loaded
  # in the future.
  NOT_READY = 1
  # Model is AVAILABLE if it is successfully loaded and ready to accept query.
  AVAILABLE = 2
  # Model is UNAVAILABLE if it is unavailable for good.
  UNAVAILABLE = 3


class BaseModelServerClient(six.with_metaclass(abc.ABCMeta, object)):
  """Common interface for all model server clients."""

  @abc.abstractmethod
  def GetModelState(self) -> ModelState:
    """Check whether the model is available for query or not.

    Returns:
      A ModelState.
    """
    pass

  @abc.abstractmethod
  def IssueRequests(self, requests: List[types.Request]) -> None:
    """Issue requests against model server.

    Args:
      requests: A list of request protos.

    Raises:
      ValueError: If request is not compatible with the client.
      grpc.RpcError: If RPC Fails.
    """
    pass
