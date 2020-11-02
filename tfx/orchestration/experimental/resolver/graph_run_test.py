# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.resolver.graph_run."""

import os
import time

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.resolver import graph_run
from tfx.orchestration.experimental.resolver import ops  # pylint: disable=unused-import
from tfx.orchestration.portable.mlmd import event_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import property_utils

from google.protobuf import message
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_MLMD_TYPE_TO_PYTHON_TYPE = {
    metadata_store_pb2.INT: int,
    metadata_store_pb2.DOUBLE: float,
    metadata_store_pb2.STRING: str
}


class _TestMixins(tf.test.TestCase):
  _store: mlmd.MetadataStore

  def setup_metadata_handler(self):
    conn_config = metadata.sqlite_metadata_connection_config(
        os.path.join(self.create_tempdir('mlmd'), 'metadata.db'))
    metadata_handler = self.enter_context(metadata.Metadata(conn_config))
    self._store = metadata_handler.store
    self._metadata_handler = metadata_handler

  def register_fake_mlmd_types(self):
    self._artifact_types = {}
    for artifact_type_name in ('A', 'B', 'C'):
      artifact_type = metadata_store_pb2.ArtifactType(
          name=artifact_type_name,
          properties={
              'i': metadata_store_pb2.INT,
              'f': metadata_store_pb2.DOUBLE,
              's': metadata_store_pb2.STRING
          })
      artifact_type.id = self._store.put_artifact_type(artifact_type)
      self._artifact_types[artifact_type.name] = artifact_type
    self._execution_types = {}
    for execution_type_name in ('X', 'Y', 'Z'):
      execution_type = metadata_store_pb2.ExecutionType(
          name=execution_type_name,
          properties={
              # component_id is used for testing "is_consumed_by_component"
              # operator.
              'component_id': metadata_store_pb2.STRING,
              'i': metadata_store_pb2.INT,
              'f': metadata_store_pb2.DOUBLE,
              's': metadata_store_pb2.STRING
          })
      execution_type.id = self._store.put_execution_type(execution_type)
      self._execution_types[execution_type.name] = execution_type

  def put_artifact(self, artifact_type_name, **props):
    artifact_type = self._artifact_types[artifact_type_name]
    artifact = metadata_store_pb2.Artifact()
    artifact.type_id = artifact_type.id
    props_proxy = property_utils.PropertyMapProxy(
        artifact.properties,
        schema={
            key: _MLMD_TYPE_TO_PYTHON_TYPE[mlmd_type]
            for key, mlmd_type in artifact_type.properties.items()
        })
    custom_props_proxy = property_utils.PropertyMapProxy(
        artifact.custom_properties)
    for key, value in props.items():
      if key in artifact_type.properties:
        props_proxy[key] = value
      else:
        custom_props_proxy[key] = value
    [artifact_id] = self._store.put_artifacts([artifact])
    [result] = self._store.get_artifacts_by_id([artifact_id])
    return result

  def put_execution(self, execution_type_name, input_artifacts,
                    output_artifacts, **props):
    execution_type = self._execution_types[execution_type_name]
    execution = metadata_store_pb2.Execution()
    execution.type_id = execution_type.id
    props_proxy = property_utils.PropertyMapProxy(
        execution.properties,
        schema={
            key: _MLMD_TYPE_TO_PYTHON_TYPE[mlmd_type]
            for key, mlmd_type in execution_type.properties.items()
        })
    custom_props_proxy = property_utils.PropertyMapProxy(
        execution.custom_properties)
    for key, value in props.items():
      if key in execution_type.properties:
        props_proxy[key] = value
      else:
        custom_props_proxy[key] = value
    artifact_and_events = []
    for key, artifacts in input_artifacts.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(
            metadata_store_pb2.Event.INPUT,
            key=key,
            index=i,
            artifact_id=artifact.id)
        artifact_and_events.append((artifact, event))
    for key, artifacts in output_artifacts.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(
            metadata_store_pb2.Event.OUTPUT,
            key=key,
            index=i,
            artifact_id=artifact.id)
        artifact_and_events.append((artifact, event))
    [execution_id] = self._store.put_execution(
        execution=execution,
        artifact_and_events=artifact_and_events,
        contexts=())
    [result] = self._store.get_executions_by_id([execution_id])
    return result

  def make_named_inputs(self, **kwargs):
    return {
        key: self._convert_to_node_input(value)
        for key, value in kwargs.items()
    }

  def _convert_to_node_input(self, value):
    r = pipeline_pb2.ResolverConfig
    if isinstance(value, r.NodeInput):
      return value
    elif isinstance(value, int):
      return r.NodeInput(int_value=value)
    elif isinstance(value, str):
      return r.NodeInput(string_value=value)
    elif isinstance(value, float):
      return r.NodeInput(float_value=value)
    elif isinstance(value, message.Message):
      return r.NodeInput(proto_value=value)
    elif isinstance(value, list):
      return r.NodeInput(
          input_list=r.NodeInput.InputList(
              values=[self._convert_to_node_input(v)
                      for v in value]))
    elif isinstance(value, dict):
      return r.NodeInput(
          input_map=r.NodeInput.InputMap(
              values={k: self._convert_to_node_input(v)
                      for k, v in value.items()}))
    raise TypeError('Unknown type {}'.format(type(value)))


class GraphRunTest(_TestMixins):

  def setUp(self):
    super().setUp()
    self.setup_metadata_handler()
    self.register_fake_mlmd_types()

  def sleep_millis(self, t: int):
    time.sleep(t * 0.001)

  def test_latest_n(self):
    r = pipeline_pb2.ResolverConfig
    artifacts = []
    for _ in range(10):
      artifacts.append(self.put_artifact('A'))
      self.sleep_millis(1)  # Wait until create timestamp change.
    channel_inputs = {
        'a': artifacts
    }
    nodes = [
        r.NodeDef(
            node_id=1,
            op_name='resolve_channel_inputs',
            named_inputs=self.make_named_inputs(
                key='a'
            )),
        r.NodeDef(
            node_id=2,
            op_name='order_by',
            named_inputs=self.make_named_inputs(
                items=r.NodeInput(node_output=1),
                criteria=[
                    'create_time_since_epoch DESC'
                ]
            )),
        r.NodeDef(
            node_id=3,
            op_name='head',
            named_inputs=self.make_named_inputs(
                items=r.NodeInput(node_output=2),
                n=3
            )),
        r.NodeDef(
            node_id=4,
            op_name='trigger_at_least',
            named_inputs=self.make_named_inputs(
                items=r.NodeInput(node_output=3),
                threshold=3
            )),
    ]
    run = graph_run.GraphRun(
        graph_def=r.ResolverGraphDef(
            nodes=nodes,
            sink={'a': 4}
        ),
        metadata_store=self._store,
        channel_inputs=channel_inputs)
    result = run.resolved_mlmd_artifacts()

    self.assertLen(result, 1)
    self.assertEqual(
        result[0],
        {
            'a': [artifacts[-1], artifacts[-2], artifacts[-3]]
        })


if __name__ == '__main__':
  tf.test.main()
