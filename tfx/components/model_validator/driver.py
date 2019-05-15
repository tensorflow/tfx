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

import tensorflow as tf
from typing import Any, Dict, List, Optional, Text, Tuple
from tfx.components.base import base_driver
from tfx.utils import types


class Driver(base_driver.BaseDriver):
  """Custom driver for model validator."""

  def _fetch_last_blessed_model(self):
    """Fetch last blessed model in metadata based on span."""
    # TODO(b/122970393): This is a temporary solution since ML metadata not
    # support get artifacts by type.
    previous_blessed_models = [
        x for x in self._metadata_handler.get_all_artifacts()
        if (x.properties['type_name'].string_value == 'ModelBlessingPath' and
            x.custom_properties['blessed'].int_value == 1)
    ]
    if previous_blessed_models:
      last_blessed_model = max(
          previous_blessed_models, key=lambda m: m.properties['span'].int_value)
      return (
          last_blessed_model.custom_properties['current_model'].string_value,
          last_blessed_model.custom_properties['current_model_id'].int_value)
    else:
      return (None, None)

  def prepare_execution(
      self,
      input_dict,
      output_dict,
      exec_properties,
      driver_options,
  ):
    """Extends BaseDriver by resolving last blessed model."""
    execution_decision = self._default_caching_handling(
        input_dict, output_dict, exec_properties, driver_options)

    # If current model isn't blessed before (no caching).
    if execution_decision.execution_id:
      (execution_decision.exec_properties['blessed_model'],
       execution_decision.exec_properties['blessed_model_id']
      ) = self._fetch_last_blessed_model()
      tf.logging.info('Resolved last blessed model {}'.format(
          execution_decision.exec_properties['blessed_model']))

    return execution_decision
