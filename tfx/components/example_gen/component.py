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

import absl

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import FileBasedExampleGenSpec
from tfx.types.standard_component_specs import QueryBasedExampleGenSpec


class _QueryBasedExampleGen(base_component.BaseComponent):
  """A TFX component to ingest examples from a file system.

  The _QueryBasedExampleGen component can be extended to ingest examples from
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
    """Construct an QueryBasedExampleGen component.

    Args:
      input_config: An
        [example_gen_pb2.Input](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing input configuration. If any field is provided as a
        RuntimeParameter, input_config should be constructed as a dict with the
        same field names as Input proto message. _required_
      output_config: An
        [example_gen_pb2.Output](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing output configuration. If unset, the default splits
        will be labeled as 'train' and 'eval' with a distribution ratio of 2:1.
        If any field is provided as a RuntimeParameter, output_config should be
        constructed as a dict with the same field names as Output proto message.
      custom_config: An
        [example_gen_pb2.CustomConfig](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing custom configuration for ExampleGen. If any field
          is provided as a RuntimeParameter, output_config should be
          constructed as a dict.
      example_artifacts: Channel of `standard_artifacts.Examples` for output
        train and eval examples.
      instance_name: Optional unique instance name. Required only if multiple
        ExampleGen components are declared in the same pipeline.
    """
    # Configure outputs.
    output_config = output_config or utils.make_default_output_config(
        input_config)
    if not example_artifacts:
      artifact = standard_artifacts.Examples()
      artifact.split_names = artifact_utils.encode_split_names(
          utils.generate_output_split_names(input_config, output_config))
      example_artifacts = channel_utils.as_channel([artifact])
    spec = QueryBasedExampleGenSpec(
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        examples=example_artifacts)
    super(_QueryBasedExampleGen, self).__init__(
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
  from tfx.utils.dsl_utils import external_input

  _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
  _data_root = os.path.join(_taxi_root, 'data', 'simple')
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = FileBasedExampleGen(input=external_input(_data_root))
  ```
  """

  SPEC_CLASS = FileBasedExampleGenSpec
  # EXECUTOR_SPEC should be overridden by subclasses.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)
  DRIVER_CLASS = driver.Driver

  def __init__(
      self,
      input: types.Channel = None,  # pylint: disable=redefined-builtin
      input_config: Optional[Union[example_gen_pb2.Input, Dict[Text,
                                                               Any]]] = None,
      output_config: Optional[Union[example_gen_pb2.Output, Dict[Text,
                                                                 Any]]] = None,
      custom_config: Optional[Union[example_gen_pb2.CustomConfig,
                                    Dict[Text, Any]]] = None,
      example_artifacts: Optional[types.Channel] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      input_base: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a FileBasedExampleGen component.

    Args:
      input: A Channel of type `standard_artifacts.ExternalArtifact`, which
        includes one artifact whose uri is an external directory containing the
        data files. _required_
      input_config: An
        [`example_gen_pb2.Input`](https://github.com/tensorflow/tfx/blob/master/tfx/proto/example_gen.proto)
          instance, providing input configuration. If unset, the files under
          input_base will be treated as a single dataset.
      output_config: An example_gen_pb2.Output instance, providing the output
        configuration. If unset, default splits will be 'train' and
        'eval' with size 2:1.
      custom_config: An optional example_gen_pb2.CustomConfig instance,
        providing custom configuration for executor.
      example_artifacts: Channel of 'ExamplesPath' for output train and eval
        examples.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor spec specified in the component attribute.
      input_base: Backwards compatibility alias for the 'input' argument.
      instance_name: Optional unique instance name. Required only if multiple
        ExampleGen components are declared in the same pipeline.  Either
        `input_base` or `input` must be present in the input arguments.
    """
    if input_base:
      absl.logging.warning(
          'The "input_base" argument to the ExampleGen component has '
          'been renamed to "input" and is deprecated. Please update your '
          'usage as support for this argument will be removed soon.')
      input = input_base
    # Configure inputs and outputs.
    input_config = input_config or utils.make_default_input_config()
    output_config = output_config or utils.make_default_output_config(
        input_config)
    if not example_artifacts:
      artifact = standard_artifacts.Examples()
      artifact.split_names = artifact_utils.encode_split_names(
          utils.generate_output_split_names(input_config, output_config))
      example_artifacts = channel_utils.as_channel([artifact])
    spec = FileBasedExampleGenSpec(
        input=input,
        input_config=input_config,
        output_config=output_config,
        custom_config=custom_config,
        examples=example_artifacts)
    super(FileBasedExampleGen, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
        instance_name=instance_name)
