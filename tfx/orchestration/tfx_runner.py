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
"""Definition of TFX runner base class."""

import abc
from typing import Any, Optional

import six

from tfx.orchestration.config import pipeline_config
from tfx.utils import doc_controls


class TfxRunner(six.with_metaclass(abc.ABCMeta, object)):
  """Base runner class for TFX.

  This is the base class for every TFX runner.
  """

  def __init__(self, config: Optional[pipeline_config.PipelineConfig] = None):
    """Initializes a TfxRunner instance.

    Args:
      config: Optional pipeline config for customizing the launching
        of each component.
    """
    self._config = config or pipeline_config.PipelineConfig()

  @abc.abstractmethod
  def run(self, pipeline) -> Optional[Any]:
    """Runs logical TFX pipeline on specific platform.

    Args:
      pipeline: logical TFX pipeline definition.

    Returns:
      Platform-specific object.
    """
    pass

  @property
  @doc_controls.do_not_doc_in_subclasses
  def config(self) -> pipeline_config.PipelineConfig:
    return self._config
