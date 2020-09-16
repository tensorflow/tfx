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
"""TFX DataViewProvider component executor."""
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import udf_utils
from tfx.components.util import value_utils
from tfx_bsl.coders import tf_graph_record_decoder

# Keys for exec_properties dict.
_CREATE_DECODER_FUNC_KEY = 'create_decoder_func'

# Keys for output_dict
_DATA_VIEW_KEY = 'data_view'


class TfGraphDataViewProviderExecutor(base_executor.BaseExecutor):
  """Executor for TfGraphDataViewProvider."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)
    create_decoder_func = udf_utils.get_fn(exec_properties,
                                           _CREATE_DECODER_FUNC_KEY)
    tf_graph_record_decoder.save_decoder(
        create_decoder_func(),
        value_utils.GetSoleValue(output_dict, _DATA_VIEW_KEY).uri)
