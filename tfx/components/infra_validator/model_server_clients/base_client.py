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

import abc
import time
from typing import List

from absl import logging
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import types


class BaseModelServerClient(abc.ABC):
  """Common interface for all model server clients."""

  @abc.abstractmethod
  def _GetServingStatus(self) -> types.ModelServingStatus:
    """Check whether the model is available for query or not.

    Returns:
      A ModelServingStatus.
    """
    pass

  def WaitUntilModelLoaded(self, deadline: float,
                           polling_interval_sec: int) -> None:
    """Wait until model is loaded and available.

    Args:
      deadline: A deadline time in UTC timestamp (in seconds).
      polling_interval_sec: GetServingStatus() polling interval.

    Raises:
      DeadlineExceeded: When deadline exceeded before model is ready.
      ValidationFailed: If validation failed explicitly.
    """
    while time.time() < deadline:
      status = self._GetServingStatus()
      if status == types.ModelServingStatus.NOT_READY:
        logging.log_every_n_seconds(
            level=logging.INFO,
            n_seconds=10,
            msg='Waiting for model to be loaded...')
        time.sleep(polling_interval_sec)
        continue
      elif status == types.ModelServingStatus.UNAVAILABLE:
        raise error_types.ValidationFailed(
            'Model server failed to load the model.')
      else:
        logging.info('Model is successfully loaded.')
        return

    raise error_types.DeadlineExceeded(
        'Deadline exceeded while waiting the model to be loaded.')

  @abc.abstractmethod
  def _SendRequest(self, request: types.Request) -> None:
    """Send a request to the model server.

    Args:
      request: A request proto.
    """
    pass

  def SendRequests(self, requests: List[types.Request]) -> None:
    """Send requests to the model server.

    Args:
      requests: A list of request protos.

    Raises:
      ValidationFailed: If error occurred while sending requests.
    """
    for r in requests:
      try:
        self._SendRequest(r)
      except Exception as e:  # pylint: disable=broad-except
        raise error_types.ValidationFailed(
            f'Model server failed to respond to the request {r}') from e
