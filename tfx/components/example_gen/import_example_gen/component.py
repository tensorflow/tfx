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
"""TFX ImportExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen.import_example_gen import executor
from tfx.proto import example_gen_pb2


class ImportExampleGen(component.FileBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX ImportExampleGen component.

  The ImportExampleGen component takes TFRecord files with TF Example data
  format, and generates train and eval examples for downsteam components.
  This component provides consistent and configurable partition, and it also
  shuffle the dataset for ML best practice.
  """

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      input: types.Channel = None,  # pylint: disable=redefined-builtin
      input_config: Optional[example_gen_pb2.Input] = None,
      output_config: Optional[example_gen_pb2.Output] = None,
      example_artifacts: Optional[types.Channel] = None,
      input_base: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct an ImportExampleGen component.

    Args:
      input: A Channel of 'ExternalPath' type, which includes one artifact
        whose uri is an external directory with TFRecord files inside
        (required).
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      input_base: Backwards compatibility alias for the 'input' argument.
      instance_name: Optional unique instance name. Necessary if multiple
        ImportExampleGen components are declared in the same pipeline.
    """
    super(ImportExampleGen, self).__init__(
        input=input,
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        input_base=input_base,
        instance_name=instance_name)
