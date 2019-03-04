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
"""TFX BigQueryExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.example_gen.big_query_example_gen import executor
from tfx.utils import channel
from tfx.utils import types


class BigQueryExampleGen(base_component.BaseComponent):
  """Official TFX BigQueryExampleGen component.

  The BigQuery examplegen component takes a query, and generates train
  and eval examples for downsteam components.

  Args:
    query: BigQuery sql string.
    name: Optional unique name. Necessary if multiple BigQueryExampleGen
      components are declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - examples: A channel of 'ExamplesPath' with train and eval examples.
  """

  def __init__(self,
               query: Text,
               name: Optional[Text] = None,
               outputs: Optional[base_component.ComponentOutputs] = None):
    component_name = 'BigQueryExampleGen'
    input_dict = {}
    exec_properties = {'query': query}
    super(BigQueryExampleGen, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self) -> base_component.ComponentOutputs:
    """Creates outputs for BigQueryExampleGen.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    output_artifact_collection = [
        types.TfxType('ExamplesPath', split=split)
        for split in types.DEFAULT_EXAMPLE_SPLITS
    ]
    return base_component.ComponentOutputs({
        'examples':
            channel.Channel(
                type_name='ExamplesPath',
                static_artifact_collection=output_artifact_collection)
    })

  def _type_check(self, input_dict: Dict[Text, channel.Channel],
                  exec_properties: Dict[Text, Any]) -> None:
    pass
