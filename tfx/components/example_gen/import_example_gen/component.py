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

from typing import Optional, Union

from tfx.components.example_gen import component
from tfx.components.example_gen.import_example_gen import executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


class ImportExampleGen(component.FileBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX ImportExampleGen component.

  The ImportExampleGen component takes TFRecord files with TF Example data
  format, and generates train and eval examples for downstream components.
  This component provides consistent and configurable partition, and it also
  shuffle the dataset for ML best practice.

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(
      self,
      input_base: Optional[str] = None,
      input_config: Optional[Union[example_gen_pb2.Input,
                                   data_types.RuntimeParameter]] = None,
      output_config: Optional[Union[example_gen_pb2.Output,
                                    data_types.RuntimeParameter]] = None,
      range_config: Optional[Union[range_config_pb2.RangeConfig,
                                   data_types.RuntimeParameter]] = None,
      payload_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE):
    """Construct an ImportExampleGen component.

    Args:
      input_base: an external directory containing the TFRecord files.
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
      payload_format: Payload format of input data. Should be one of
        example_gen_pb2.PayloadFormat enum. Note that payload format of output
        data is the same as input.
    """
    super().__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        range_config=range_config,
        output_data_format=payload_format)
