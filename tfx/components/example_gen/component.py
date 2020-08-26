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
from absl import logging

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import FileBasedExampleGenSpec
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec


class QueryBasedExampleGen(base_component.BaseComponent):
  """A TFX component to ingest examples from query based systems.

  The QueryBasedExampleGen component can be extended to ingest examples from
  query based systems such as Presto or Bigquery. The component will also
  convert the input data into
  tf.record](https://www.tensorflow.org/tutorials/load_data/tf_records)
  and generate train and eval example splits for downsteam components.

  ## Example
  ```
  _query = "SELECT * FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`"
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = BigQueryExampleGen(query=_query)
  ```
  """

  SPEC_CLASS = QueryBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self,
               input_config: Union[example_gen_pb2.Input, Dict[Text, Any]],
               output_config: Optional[Union[example_gen_pb2.Output,
                                             Dict[Text, Any]]] = None,
               custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                             Dict[Text, Any]]] = None,
               example_artifacts: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
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
      example_artifacts: Channel of `standard_artifacts.Examples` for output
        train and eval examples.
      instance_name: Optional unique instance name. Required only if multiple
        ExampleGen components are declared in the same pipeline.
    """
    # Configure outputs.
    output_config = output_config or utils.make_default_output_config(
        input_config)
    if not example_artifacts:
      example_artifacts = types.Channel(type=standard_artifacts.Examples)
    spec = QueryBasedExampleGenSpec(
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        examples=example_artifacts)
    super(QueryBasedExampleGen, self).__init__(
        spec=spec, instance_name=instance_name)


class FileBasedExampleGen(base_component.BaseComponent):
  """A TFX component to ingest examples from a file system.

  The FileBasedExampleGen component is an API for getting file-based records
  into TFX pipelines. It consumes external files to generate examples which will
  be used by other internal components like StatisticsGen or Trainers.  The
  component will also convert the input data into
  [tf.record](https://www.tensorflow.org/tutorials/load_data/tf_records)
  and generate train and eval example splits for downsteam components.

  ## Example
  ```
  _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
  _data_root = os.path.join(_taxi_root, 'data', 'simple')
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = FileBasedExampleGen(input_base=_data_root)
  ```
  """

  SPEC_CLASS = FileBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)
  DRIVER_CLASS = driver.Driver

  def __init__(
      self,
      # TODO(b/159467778): deprecate this, use input_base instead.
      input: Optional[types.Channel] = None,  # pylint: disable=redefined-builtin
      input_base: Optional[Text] = None,
      input_config: Optional[Union[example_gen_pb2.Input, Dict[Text,
                                                               Any]]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                    Dict[Text, Any]]] = None,
      output_data_format: Optional[int] = example_gen_pb2.FORMAT_TF_EXAMPLE,
      example_artifacts: Optional[types.Channel] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      instance_name: Optional[Text] = None):
    """Construct a FileBasedExampleGen component.

    Args:
      input: A Channel of type `standard_artifacts.ExternalArtifact`, which
        includes one artifact whose uri is an external directory containing the
        data files. (Deprecated by input_base)
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
      output_data_format: Payload format of generated data in output artifact,
        one of example_gen_pb2.PayloadFormat enum.
      example_artifacts: Channel of 'ExamplesPath' for output train and eval
        examples.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor spec specified in the component attribute.
      instance_name: Optional unique instance name. Required only if multiple
        ExampleGen components are declared in the same pipeline.
    """
    if input:
      logging.warning(
          'The "input" argument to the ExampleGen component has been '
          'deprecated by "input_base". Please update your usage as support for '
          'this argument will be removed soon.')
      input_base = artifact_utils.get_single_uri(list(input.get()))
    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = output_config or utils.make_default_output_config(
        input_config)
    if not example_artifacts:
      example_artifacts = types.Channel(type=standard_artifacts.Examples)
    spec = FileBasedExampleGenSpec(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        output_data_format=output_data_format,
        examples=example_artifacts)
    super(FileBasedExampleGen, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
        instance_name=instance_name)
