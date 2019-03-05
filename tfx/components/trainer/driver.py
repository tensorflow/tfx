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

from typing import Any, Dict, List, Text
from tfx.components.base import base_driver
from tfx.utils import types


class Driver(base_driver.BaseDriver):
  """Custom driver for Trainer."""

  def _fetch_latest_model(self):
    previous_models = [
        x for x in self._metadata_handler.get_all_artifacts()
        if x.properties['type_name'].string_value == 'ModelExportPath'
    ]
    if previous_models:
      latest_model = max(
          previous_models, key=lambda m: m.properties['span'].int_value)
      return latest_model.uri

    return None

  def prepare_execution(
      self,
      input_dict,
      output_dict,
      exec_properties,
      driver_options,
  ):
    """Extends BaseDriver.prepare_execution() for potential warm starting."""

    execution_decision = self._default_caching_handling(
        input_dict, output_dict, exec_properties, driver_options)

    # Fetch latest model dir for warms-tarting if needed.
    if execution_decision.execution_id and execution_decision.exec_properties.get(
        'warm_starting', None):
      execution_decision.exec_properties[
          'warm_start_from'] = self._fetch_latest_model()
      self._logger.debug('Model directory to warm start from: {}'.format(
          execution_decision.exec_properties['warm_start_from']))

    return execution_decision
