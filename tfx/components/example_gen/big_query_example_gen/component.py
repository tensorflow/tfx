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

from typing import Optional, Text

from tfx.components.base import base_component
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.components.example_gen.big_query_example_gen import executor
from tfx.proto import example_gen_pb2


class BigQueryExampleGen(component.ExampleGen):
  """Official TFX BigQueryExampleGen component.

  The BigQuery examplegen component takes a query, and generates train
  and eval examples for downsteam components.

  Args:
    query: BigQuery sql string, query result will be treated as a single split,
      can be overwritten by input_config.
    input_config: An example_gen_pb2.Input instance with Split.pattern as
      BigQuery sql string. If set, it overwrites the 'query' arg, and allows
      different queries per split.
    output_config: An example_gen_pb2.Output instance, providing output
      configuration. If unset, default splits will be 'train' and 'eval' with
      size 2:1.
    name: Optional unique name. Necessary if multiple BigQueryExampleGen
      components are declared in the same pipeline.
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - examples: A channel of 'ExamplesPath' with train and eval examples.
  Raises:
    RuntimeError: Only one of query and input_config should be set.
  """

  def __init__(self,
               query = None,
               input_config = None,
               output_config = None,
               name = None,
               outputs = None):
    if bool(query) == bool(input_config):
      raise RuntimeError('Only one of query and input_config should be set.')
    input_config = input_config or utils.make_default_input_config(query)
    output_config = output_config or utils.make_default_output_config(
        input_config)
    super(BigQueryExampleGen, self).__init__(
        executor=executor.Executor,
        input_base=None,
        input_config=input_config,
        output_config=output_config,
        component_name='BigQueryExampleGen',
        unique_name=name,
        outputs=outputs)
