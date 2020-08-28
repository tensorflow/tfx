# Lint as: python3
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
"""Tests for tfx.orchestration.portable.mlmd.execution_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfx.orchestration import metadata
from tfx.orchestration.portable import test_utils
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class ExecutionLibTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()

  def _generate_contexts(self, metadata_handler):
    context_spec = pipeline_pb2.NodeContexts()
    text_format.Parse(
        """
        contexts {
          type {name: 'pipeline_context'}
          name {
            field_value {string_value: 'my_pipeline'}
          }
        }
        contexts {
          type {name: 'component_context'}
          name {
            field_value {string_value: 'my_component'}
          }
        }""", context_spec)
    return context_lib.register_contexts_if_not_exists(metadata_handler,
                                                       context_spec)

  def testPrepareExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      execution_type = metadata_store_pb2.ExecutionType()
      text_format.Parse(
          """
          name: 'my_execution'
          properties {
            key: 'p2'
            value: STRING
          }
          """, execution_type)
      result = execution_lib.prepare_execution(
          m,
          execution_type,
          exec_properties={
              'p1': 1,
              'p2': '2'
          },
          state=metadata_store_pb2.Execution.COMPLETE)
      self.assertProtoEquals(
          """
          type_id: 1
          last_known_state: COMPLETE
          properties {
            key: 'p2'
            value {
              string_value: '2'
            }
          }
          custom_properties {
            key: 'p1'
            value {
              int_value: 1
            }
          }
          """, result)

  def testArtifactAndEventPairs(self):
    example = standard_artifacts.Examples()
    example.uri = 'example'
    example.id = 1

    expected_artifact = metadata_store_pb2.Artifact()
    text_format.Parse(
        """
        id: 1
        type_id: 1
        uri: 'example'""", expected_artifact)
    expected_event = metadata_store_pb2.Event()
    text_format.Parse(
        """
        path {
          steps {
            key: 'example'
          }
          steps {
            index: 0
          }
        }
        type: INPUT""", expected_event)

    with metadata.Metadata(connection_config=self._connection_config) as m:
      result = execution_lib._create_artifact_and_event_pairs(
          m, {
              'example': [example],
          }, metadata_store_pb2.Event.INPUT)

      self.assertCountEqual([(expected_artifact, expected_event)], result)

  def testPutExecutionGraph(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Prepares an input artifact. The artifact should be registered in MLMD
      # before the put_execution call.
      input_example = standard_artifacts.Examples()
      input_example.uri = 'example'
      input_example.type_id = common_utils.register_type_if_not_exist(
          m, input_example.artifact_type).id
      [input_example.id] = m.store.put_artifacts([input_example.mlmd_artifact])
      # Prepares an output artifact.
      output_model = standard_artifacts.Model()
      output_model.uri = 'model'
      execution = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          exec_properties={
              'p1': 1,
              'p2': '2'
          },
          state=metadata_store_pb2.Execution.COMPLETE)
      contexts = self._generate_contexts(m)
      execution = execution_lib.put_execution(
          m,
          execution,
          contexts,
          input_artifacts={'example': [input_example]},
          output_artifacts={'model': [output_model]})

      self.assertProtoPartiallyEquals(
          output_model.mlmd_artifact,
          m.store.get_artifacts_by_id([output_model.id])[0],
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      # Verifies edges between artifacts and execution.
      [input_event] = m.store.get_events_by_artifact_ids([input_example.id])
      self.assertEqual(input_event.execution_id, execution.id)
      self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
      [output_event] = m.store.get_events_by_artifact_ids([output_model.id])
      self.assertEqual(output_event.execution_id, execution.id)
      self.assertEqual(output_event.type, metadata_store_pb2.Event.OUTPUT)
      # Verifies edges connecting contexts and {artifacts, execution}.
      context_ids = [context.id for context in contexts]
      self.assertCountEqual(
          [c.id for c in m.store.get_contexts_by_artifact(input_example.id)],
          context_ids)
      self.assertCountEqual(
          [c.id for c in m.store.get_contexts_by_artifact(output_model.id)],
          context_ids)
      self.assertCountEqual(
          [c.id for c in m.store.get_contexts_by_execution(execution.id)],
          context_ids)


if __name__ == '__main__':
  tf.test.main()
