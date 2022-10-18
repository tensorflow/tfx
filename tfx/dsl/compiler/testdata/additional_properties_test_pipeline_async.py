# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Test pipeline using custom components for tfx.dsl.compiler.compiler."""

import os

from typing import Any, Dict, List, Optional

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_property
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2

# TODO(b/241861488): Remove safeguard once fully supported by MLMD.
artifact_property.ENABLE_PROTO_PROPERTIES = True


class StatsMetadata(Artifact):
  TYPE_NAME = 'StatsMetadata'
  PROPERTIES = {
      'timestamp': Property(type=PropertyType.PROTO
                           ),  # Expected proto type: google.protobuf.Timestamp
      'ttl': Property(type=PropertyType.PROTO
                     ),  # Expected proto type: google.protobuf.Duration
  }


class CustomProducerSpec(types.ComponentSpec):
  """ComponentSpec for Custom Producer Component."""

  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {
      'stats':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExampleStatistics),
      'stats_metadata':
          component_spec.ChannelParameter(type=StatsMetadata),
  }


class CustomConsumerSpec(types.ComponentSpec):
  """ComponentSpec for Custom Comsumer Component."""

  PARAMETERS = {}
  INPUTS = {
      'data':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExampleStatistics),
  }
  OUTPUTS = {}


class DummyExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    pass


class CustomProducer(base_component.BaseComponent):
  """Custom Producer Component."""

  SPEC_CLASS = CustomProducerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(DummyExecutor)

  def __init__(self,
               stats: Optional[types.Channel] = None,
               stats_metadata: Optional[types.Channel] = None):
    stats = stats or types.Channel(type=standard_artifacts.ExampleStatistics)
    stats.additional_properties['span'] = 42
    stats.additional_properties['split_names'] = '[\'train\', \'eval\']'
    stats.additional_custom_properties['bar'] = 'foo'
    stats.additional_custom_properties['baz'] = 0.5

    stats_metadata = stats_metadata or types.Channel(type=StatsMetadata)
    timestamp_value = pipeline_pb2.Value()
    timestamp_value.field_value.proto_value.Pack(
        timestamp_pb2.Timestamp(seconds=999999))
    stats_metadata.additional_properties['timestamp'] = timestamp_value
    ttl_value = pipeline_pb2.Value()
    ttl_value.field_value.proto_value.Pack(duration_pb2.Duration(seconds=600))

    stats_metadata.additional_properties['ttl'] = ttl_value
    spec = CustomProducerSpec(stats=stats, stats_metadata=stats_metadata)
    super().__init__(spec=spec)


class CustomConsumer(base_component.BaseComponent):
  """Custom Consumer Component."""

  SPEC_CLASS = CustomConsumerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(DummyExecutor)

  def __init__(self, data: types.Channel):
    spec = CustomConsumerSpec(data=data)
    super().__init__(spec=spec)


def create_test_pipeline():
  """Builds an example asynchronous pipeline using custom components."""
  pipeline_name = 'custom1'
  tfx_root = 'tfx_root'
  pipeline_root = os.path.join(tfx_root, 'pipelines', pipeline_name)

  custom_producer = CustomProducer()

  custom_consumer = CustomConsumer(data=custom_producer.outputs['stats'])

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          custom_producer,
          custom_consumer,
      ],
      execution_mode=pipeline.ExecutionMode.ASYNC)
