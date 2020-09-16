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
"""TFX ImportExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union
from absl import logging

from tfx import types
from tfx.components.base import executor_spec
from tfx.components.example_gen import component
from tfx.components.example_gen.import_example_gen import executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils


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
      # TODO(b/159467778): deprecate this, use input_base instead.
      input: Optional[types.Channel] = None,  # pylint: disable=redefined-builtin
      input_base: Optional[Text] = None,
      input_config: Optional[Union[example_gen_pb2.Input, Dict[Text,
                                                               Any]]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      payload_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE,
      example_artifacts: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct an ImportExampleGen component.

    Args:
       input: A Channel of type `standard_artifacts.ExternalArtifact`, which
        includes one artifact whose uri is an external directory containing the
        TFRecord files. (Deprecated by input_base)
      input_base: an external directory containing the TFRecord files.
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
      payload_format: Payload format of input data. Should be one of
        example_gen_pb2.PayloadFormat enum. Note that payload format of output
        data is the same as input.
      example_artifacts: Optional channel of 'ExamplesPath' for output train and
        eval examples.
      instance_name: Optional unique instance name. Necessary if multiple
        ImportExampleGen components are declared in the same pipeline.
    """
    if input:
      logging.warning(
          'The "input" argument to the ImportExampleGen component has been '
          'deprecated by "input_base". Please update your usage as support for '
          'this argument will be removed soon.')
      input_base = artifact_utils.get_single_uri(list(input.get()))
    super(ImportExampleGen, self).__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        output_data_format=payload_format,
        instance_name=instance_name)
