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
"""Tests for tfx.orchestration.portable.execution_publish_utils."""
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import test_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class ExecutionPublisherTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._execution_type = metadata_store_pb2.ExecutionType(name='my_ex_type')

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

  def testRegisterExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      input_example = standard_artifacts.Examples()
      execution_publish_utils.register_execution(
          m,
          self._execution_type,
          contexts,
          input_artifacts={'examples': [input_example]},
          exec_properties={
              'p1': 1,
          })
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 3
          custom_properties {
            key: 'p1'
            value {int_value: 1}
          }
          last_known_state: RUNNING
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [event] = m.store.get_events_by_execution_ids([execution.id])
      self.assertProtoPartiallyEquals(
          """
          artifact_id: 1
          execution_id: 1
          path {
            steps {
              key: 'examples'
            }
            steps {
              index: 0
            }
          }
          type: INPUT
          """,
          event,
          ignored_fields=['milliseconds_since_epoch'])
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_artifact(input_example.id)])

  def testPublishCachedExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      output_example = standard_artifacts.Examples()
      execution_publish_utils.publish_cached_execution(
          m,
          contexts,
          execution_id,
          output_artifacts={'examples': [output_example]})
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 3
          last_known_state: CACHED
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [event] = m.store.get_events_by_execution_ids([execution.id])
      self.assertProtoPartiallyEquals(
          """
          artifact_id: 1
          execution_id: 1
          path {
            steps {
              key: 'examples'
            }
            steps {
              index: 0
            }
          }
          type: OUTPUT
          """,
          event,
          ignored_fields=['milliseconds_since_epoch'])
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_artifact(output_example.id)])

  def testPublishSuccessfulExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      output_key = 'examples'
      output_example = standard_artifacts.Examples()
      executor_output = execution_result_pb2.ExecutorOutput()
      text_format.Parse(
          """
          uri: 'examples_uri'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }
          """, executor_output.output_artifacts[output_key].artifacts.add())
      execution_publish_utils.publish_succeeded_execution(
          m, execution_id, contexts, {output_key: [output_example]},
          executor_output)
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [artifact] = m.store.get_artifacts()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 4
          state: LIVE
          uri: 'examples_uri'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }""",
          artifact,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [event] = m.store.get_events_by_execution_ids([execution.id])
      self.assertProtoPartiallyEquals(
          """
          artifact_id: 1
          execution_id: 1
          path {
            steps {
              key: 'examples'
            }
            steps {
              index: 0
            }
          }
          type: OUTPUT
          """,
          event,
          ignored_fields=['milliseconds_since_epoch'])
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_artifact(output_example.id)])

  def testPublishSuccessExecutionFailNewKey(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      executor_output = execution_result_pb2.ExecutorOutput()
      executor_output.output_artifacts['new_key'].artifacts.add()

      with self.assertRaisesRegex(RuntimeError, 'contains more keys'):
        execution_publish_utils.publish_succeeded_execution(
            m, execution_id, contexts,
            {'examples': [standard_artifacts.Examples()]}, executor_output)

  def testPublishSuccessExecutionFailPartialUpdate(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      executor_output = execution_result_pb2.ExecutorOutput()
      executor_output.output_artifacts['examples'].artifacts.add()

      with self.assertRaisesRegex(RuntimeError, 'Partially update an output'):
        execution_publish_utils.publish_succeeded_execution(
            m, execution_id, contexts, {
                'examples': [
                    standard_artifacts.Examples(),
                    standard_artifacts.Examples()
                ]
            }, executor_output)

  def testPublishSuccessExecutionFailChangedType(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      executor_output = execution_result_pb2.ExecutorOutput()
      executor_output.output_artifacts['examples'].artifacts.add().type_id = 10

      with self.assertRaisesRegex(RuntimeError, 'change artifact type'):
        execution_publish_utils.publish_succeeded_execution(
            m, execution_id, contexts,
            {'examples': [standard_artifacts.Examples(),]}, executor_output)

  def testPublishFailedExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      execution_publish_utils.publish_failed_execution(m, contexts,
                                                       execution_id)
      [execution] = m.store.get_executions_by_id([execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 3
          last_known_state: FAILED
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      # No events because there is no artifact published.
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertEmpty(events)
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])


if __name__ == '__main__':
  tf.test.main()
