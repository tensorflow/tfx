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
"""TFX CsvExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen.csv_example_gen import executor
from tfx.proto import example_gen_pb2


class CsvExampleGen(component.FileBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX CsvExampleGen component.

  The csv examplegen component takes csv data, and generates train
  and eval examples for downsteam components.
  """

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      input: types.Channel = None,  # pylint: disable=redefined-builtin
      input_config: Optional[Union[example_gen_pb2.Input, Dict[Text,
                                                               Any]]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      example_artifacts: Optional[types.Channel] = None,
      input_base: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a CsvExampleGen component.

    Args:
      input: A Channel of type `standard_artifacts.ExternalArtifact`, which
        includes one artifact whose uri is an external directory containing csv
        files (required).
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split. If any field is provided as a RuntimeParameter,
        input_config should be constructed as a dict with the same field names
        as Input proto message.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1. If any field is provided as a RuntimeParameter,
        output_config should be constructed as a dict with the same field names
        as Output proto message.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      input_base: Backwards compatibility alias for the 'input' argument.
      instance_name: Optional unique instance name. Necessary if multiple
        CsvExampleGen components are declared in the same pipeline.
    """
    super(CsvExampleGen, self).__init__(
        input=input,
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        input_base=input_base,
        instance_name=instance_name)
