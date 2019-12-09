# Lint as: python2, python3
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
"""TFX Trainer Driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text

import absl

from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.types import standard_artifacts


class Driver(base_driver.BaseDriver):
  """Custom driver for Trainer."""

  def _fetch_latest_model(self):
    previous_models = self._metadata_handler.get_artifacts_by_type(
        standard_artifacts.Model.TYPE_NAME)
    if previous_models:
      # TODO(b/138845899): consider use span instead of id.
      latest_model = max(previous_models, key=lambda artifact: artifact.id)
      return latest_model.uri

    return None

  def resolve_exec_properties(
      self,
      exec_properties: Dict[Text, Any],
      pipeline_info: data_types.PipelineInfo,  # pylint: disable=unused-argument
      component_info: data_types.ComponentInfo
  ) -> Dict[Text, Any]:
    """Overrides BaseDriver.resolve_exec_properties()."""
    if exec_properties.get('warm_starting', None):
      exec_properties['warm_start_from'] = self._fetch_latest_model()
      absl.logging.debug('Model directory to warm start from: {}'.format(
          exec_properties['warm_start_from']))
    return exec_properties
