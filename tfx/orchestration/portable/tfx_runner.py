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
"""Definition of TFX runner base class."""

import abc
from typing import Any, Optional, Union

from tfx.orchestration import pipeline as pipeline_py
from tfx.proto.orchestration import pipeline_pb2


class TfxRunner(metaclass=abc.ABCMeta):
  """Base runner class for TFX.

  This is the base class for every TFX runner.
  """

  @abc.abstractmethod
  def run(
      self, pipeline: Union[pipeline_pb2.Pipeline,
                            pipeline_py.Pipeline]) -> Optional[Any]:
    """Runs a TFX pipeline on a specific platform.

    Args:
      pipeline: a pipeline_pb2.Pipeline message or pipeline.Pipeline instance
        representing a pipeline definition.

    Returns:
      Optional platform-specific object.
    """
    pass
