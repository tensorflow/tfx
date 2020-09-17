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
"""Generic TFX model validator custom driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Tuple

import absl

from tfx.dsl.components.base import base_driver
from tfx.orchestration import data_types


class Driver(base_driver.BaseDriver):
  """Custom driver for model validator."""

  def _fetch_last_blessed_model(
      self,
      pipeline_name: Text,
      component_id: Text,
  ) -> Tuple[Optional[Text], Optional[int]]:
    """Fetch last blessed model in metadata based on span."""
    previous_blessed_models = []
    for a in self._metadata_handler.get_artifacts_by_type('ModelBlessing'):
      # TODO(ccy): get pipeline name from MLMD context.
      if 'pipeline_name' in a.properties:
        p = a.properties['pipeline_name'].string_value
      else:
        p = a.custom_properties['pipeline_name'].string_value
      if (p == pipeline_name and
          a.custom_properties['blessed'].int_value == 1 and
          a.custom_properties['component_id'].string_value == component_id):
        previous_blessed_models.append(a)

    if previous_blessed_models:
      # TODO(b/138845899): consider use span instead of id.
      last_blessed_model = max(
          previous_blessed_models, key=lambda artifact: artifact.id)
      return (
          last_blessed_model.custom_properties['current_model'].string_value,
          last_blessed_model.custom_properties['current_model_id'].int_value)
    else:
      return None, None

  # pyformat: disable
  def resolve_exec_properties(
      self, exec_properties: Dict[Text, Any],
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo) -> Dict[Text, Any]:
    # pyformat: enable
    """Overrides BaseDriver.resolve_exec_properties()."""
    (exec_properties['blessed_model'],
     exec_properties['blessed_model_id']) = self._fetch_last_blessed_model(
         pipeline_info.pipeline_name, component_info.component_id)
    exec_properties['current_component_id'] = component_info.component_id
    absl.logging.info('Resolved last blessed model {}'.format(
        exec_properties['blessed_model']))
    return exec_properties
