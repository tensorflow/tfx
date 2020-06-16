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
"""TFX PrestoExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from presto_component import executor
from proto import presto_config_pb2

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2


class PrestoExampleGen(component.QueryBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX PrestoExampleGen component.

  The Presto examplegen component takes a query, connection client
  configuration, and generates train and eval examples for downsteam components.
  """
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               conn_config: presto_config_pb2.PrestoConnConfig,
               query: Optional[Text] = None,
               input_config: Optional[example_gen_pb2.Input] = None,
               output_config: Optional[example_gen_pb2.Output] = None,
               example_artifacts: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Constructs a PrestoExampleGen component.

    Args:
      conn_config: Parameters for Presto connection client.
      query: Presto sql string, query result will be treated as a single split,
        can be overwritten by input_config.
      input_config: An example_gen_pb2.Input instance with Split.pattern as
        Presto sql string. If set, it overwrites the 'query' arg, and allows
        different queries per split.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      instance_name: Optional unique instance name. Necessary if multiple
        PrestoExampleGen components are declared in the same pipeline.

    Raises:
      RuntimeError: Only one of query and input_config should be set. Or
      required host field in connection_config should be set.
    """
    if bool(query) == bool(input_config):
      raise RuntimeError('Exactly one of query and input_config should be set.')
    if not bool(conn_config.host):
      raise RuntimeError(
          'Required host field in connection config should be set.')

    input_config = input_config or utils.make_default_input_config(query)

    packed_custom_config = example_gen_pb2.CustomConfig()
    packed_custom_config.custom_config.Pack(conn_config)

    output_config = output_config or utils.make_default_output_config(
        input_config)

    super(PrestoExampleGen, self).__init__(
        input_config=input_config,
        output_config=output_config,
        custom_config=packed_custom_config,
        example_artifacts=example_artifacts,
        instance_name=instance_name)
