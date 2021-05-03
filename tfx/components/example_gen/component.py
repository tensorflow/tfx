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
"""TFX ExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from tfx import types
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import FileBasedExampleGenSpec
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec


class QueryBasedExampleGen(base_beam_component.BaseBeamComponent):
  """A TFX component to ingest examples from query based systems.

  The QueryBasedExampleGen component can be extended to ingest examples from
  query based systems such as Presto or Bigquery. The component will also
  convert the input data into
  tf.record](https://www.tensorflow.org/tutorials/load_data/tf_records)
  and generate train and eval example splits for downstream components.

  ## Example
  ```
  _query = "SELECT * FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`"
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=_query)
  ```

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  SPEC_CLASS = QueryBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
      base_beam_executor.BaseBeamExecutor)
  DRIVER_CLASS = driver.QueryBasedDriver

  def __init__(
      self,
      input_config: Union[example_gen_pb2.Input, Dict[Text, Any]],
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                    Dict[Text, Any]]] = None,
      output_data_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE):
    """Construct a QueryBasedExampleGen component.

    Args:
      input_config: An
        [example_gen_pb2.Input](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing input configuration. If any field is provided as a
          RuntimeParameter, input_config should be constructed as a dict with
          the same field names as Input proto message. _required_
      output_config: An
        [example_gen_pb2.Output](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing output configuration. If unset, the default splits
        will be labeled as 'train' and 'eval' with a distribution ratio of 2:1.
          If any field is provided as a RuntimeParameter, output_config should
          be constructed as a dict with the same field names as Output proto
          message.
      custom_config: An
        [example_gen_pb2.CustomConfig](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing custom configuration for ExampleGen. If any field
          is provided as a RuntimeParameter, output_config should be constructed
          as a dict.
      output_data_format: Payload format of generated data in output artifact,
        one of example_gen_pb2.PayloadFormat enum.

    Raises:
      ValueError: The output_data_format value must be defined in the
        example_gen_pb2.PayloadFormat proto.
    """
    # Configure outputs.
    output_config = output_config or utils.make_default_output_config(
        input_config)
    example_artifacts = types.Channel(type=standard_artifacts.Examples)
    if output_data_format not in example_gen_pb2.PayloadFormat.values():
      raise ValueError('The value of output_data_format must be defined in'
                       'the example_gen_pb2.PayloadFormat proto.')

    spec = QueryBasedExampleGenSpec(
        input_config=input_config,
        output_config=output_config,
        output_data_format=output_data_format,
        custom_config=custom_config,
        examples=example_artifacts)
    super(QueryBasedExampleGen, self).__init__(spec=spec)


class FileBasedExampleGen(base_beam_component.BaseBeamComponent):
  """A TFX component to ingest examples from a file system.

  The FileBasedExampleGen component is an API for getting file-based records
  into TFX pipelines. It consumes external files to generate examples which will
  be used by other internal components like StatisticsGen or Trainers.  The
  component will also convert the input data into
  [tf.record](https://www.tensorflow.org/tutorials/load_data/tf_records)
  and generate train and eval example splits for downstream components.

  ## Example
  ```
  _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
  _data_root = os.path.join(_taxi_root, 'data', 'simple')
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = FileBasedExampleGen(input_base=_data_root)
  ```

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  SPEC_CLASS = FileBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
      base_beam_executor.BaseBeamExecutor)
  DRIVER_CLASS = driver.FileBasedDriver

  def __init__(
      self,
      input_base: Optional[Text] = None,
      input_config: Optional[Union[example_gen_pb2.Input, Dict[Text,
                                                               Any]]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                    Dict[Text, Any]]] = None,
      range_config: Optional[Union[range_config_pb2.RangeConfig,
                                   Dict[Text, Any]]] = None,
      output_data_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
    """Construct a FileBasedExampleGen component.

    Args:
      input_base: an external directory containing the data files.
      input_config: An
        [`example_gen_pb2.Input`](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing input configuration. If unset, input files will be
          treated as a single split.
      output_config: An example_gen_pb2.Output instance, providing the output
        configuration. If unset, default splits will be 'train' and
        'eval' with size 2:1.
      custom_config: An optional example_gen_pb2.CustomConfig instance,
        providing custom configuration for executor.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
      output_data_format: Payload format of generated data in output artifact,
        one of example_gen_pb2.PayloadFormat enum.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor spec specified in the component attribute.
    """
    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = output_config or utils.make_default_output_config(
        input_config)
    example_artifacts = types.Channel(type=standard_artifacts.Examples)
    spec = FileBasedExampleGenSpec(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        range_config=range_config,
        output_data_format=output_data_format,
        examples=example_artifacts)
    super(FileBasedExampleGen, self).__init__(
        spec=spec, custom_executor_spec=custom_executor_spec)
