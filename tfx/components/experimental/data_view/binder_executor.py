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
"""TFX DataViewBinder component executor."""
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.experimental.data_view import constants
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils


# Keys for input_dict.
_INPUT_EXAMPLES_KEY = 'input_examples'
_DATA_VIEW_KEY = 'data_view'

# Keys for output_dict.
_OUTPUT_EXAMPLES_KEY = 'output_examples'


class DataViewBinderExecutor(base_executor.BaseExecutor):
  """Executor for DataViewBinder."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)

    data_view_artifact = artifact_utils.get_single_instance(
        input_dict.get(_DATA_VIEW_KEY))
    input_examples_artifact = artifact_utils.get_single_instance(
        input_dict.get(_INPUT_EXAMPLES_KEY))
    output_examples_artifact = artifact_utils.get_single_instance(
        output_dict.get(_OUTPUT_EXAMPLES_KEY, []))

    # The output artifact shares the URI and all other properties with the
    # input, with the following additional custom properties added.
    output_examples_artifact.copy_from(input_examples_artifact)
    output_examples_artifact.set_int_custom_property(
        constants.DATA_VIEW_CREATE_TIME_KEY,
        data_view_artifact.mlmd_artifact.create_time_since_epoch)
    output_examples_artifact.set_string_custom_property(
        constants.DATA_VIEW_URI_PROPERTY_KEY, data_view_artifact.uri)
