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
from absl.testing import parameterized
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class ExecutionPublisherTest(test_case_utils.TfxTest, parameterized.TestCase):

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
    return context_lib.prepare_contexts(metadata_handler, context_spec)

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
          custom_properties {
            key: 'p1'
            value {int_value: 1}
          }
          last_known_state: RUNNING
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
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
          last_known_state: CACHED
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
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
      output_example.uri = '/examples_uri'
      executor_output = execution_result_pb2.ExecutorOutput()
      text_format.Parse(
          """
          uri: '/examples_uri'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }
          """, executor_output.output_artifacts[output_key].artifacts.add())
      output_dict = execution_publish_utils.publish_succeeded_execution(
          m, execution_id, contexts, {output_key: [output_example]},
          executor_output)
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])
      [artifact] = m.store.get_artifacts()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          state: LIVE
          uri: '/examples_uri'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }""",
          artifact,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
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
      for artifact_list in output_dict.values():
        for output_example in artifact_list:
          self.assertCountEqual([c.id for c in contexts], [
              c.id for c in m.store.get_contexts_by_artifact(output_example.id)
          ])

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

  def testPublishSuccessExecutionExecutorEditedOutputDict(self):
    # There is one artifact in the system provided output_dict, while there are
    # two artifacts in executor output. We expect that the two updated artifacts
    # with their updated properties are what is published.
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id

      output_example = standard_artifacts.Examples()
      output_example.uri = '/original_path'
      # The executor output overrides this property value
      output_example.set_int_custom_property('prop', 0)

      executor_output = execution_result_pb2.ExecutorOutput()
      output_key = 'examples'
      text_format.Parse(
          """
          uri: '/original_path/subdir_1'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }
          """, executor_output.output_artifacts[output_key].artifacts.add())
      text_format.Parse(
          """
          uri: '/original_path/subdir_2'
          custom_properties {
            key: 'prop'
            value {int_value: 2}
          }
          """, executor_output.output_artifacts[output_key].artifacts.add())

      output_dict = execution_publish_utils.publish_succeeded_execution(
          m, execution_id, contexts, {output_key: [output_example]},
          executor_output)
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])
      artifacts = m.store.get_artifacts()
      self.assertLen(artifacts, 2)
      self.assertProtoPartiallyEquals(
          """
          id: 1
          state: LIVE
          uri: '/original_path/subdir_1'
          custom_properties {
            key: 'prop'
            value {int_value: 1}
          }""",
          artifacts[0],
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      self.assertProtoPartiallyEquals(
          """
          id: 2
          state: LIVE
          uri: '/original_path/subdir_2'
          custom_properties {
            key: 'prop'
            value {int_value: 2}
          }""",
          artifacts[1],
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertLen(events, 2)
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
          events[0],
          ignored_fields=['milliseconds_since_epoch'])
      self.assertProtoPartiallyEquals(
          """
          artifact_id: 2
          execution_id: 1
          path {
            steps {
              key: 'examples'
            }
            steps {
              index: 1
            }
          }
          type: OUTPUT
          """,
          events[1],
          ignored_fields=['milliseconds_since_epoch'])
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])
      for artifact_list in output_dict.values():
        for output_example in artifact_list:
          self.assertCountEqual([c.id for c in contexts], [
              c.id for c in m.store.get_contexts_by_artifact(output_example.id)
          ])

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

  @parameterized.named_parameters(
      # Not direct sub-dir of the original uri
      dict(
          testcase_name='TooManyDirLayers', invalid_uri='/my/original_uri/1/1'),
      # Identical to the original uri
      dict(
          testcase_name='IdenticalToTheOriginal',
          invalid_uri='/my/original_uri'),
      # Identical to the original uri
      dict(testcase_name='ParentDirChanged', invalid_uri='/my/other_uri/1'),
  )
  def testPublishSuccessExecutionFailInvalidUri(self, invalid_uri):
    output_example = standard_artifacts.Examples()
    output_example.uri = '/my/original_uri'
    output_dict = {'examples': [output_example]}
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      executor_output = execution_result_pb2.ExecutorOutput()
      system_generated_artifact = executor_output.output_artifacts[
          'examples'].artifacts.add()
      system_generated_artifact.uri = '/my/original_uri/0'
      new_artifact = executor_output.output_artifacts['examples'].artifacts.add(
      )
      new_artifact.uri = invalid_uri

      with self.assertRaisesRegex(
          RuntimeError,
          'When there are multiple artifacts to publish, their URIs should be '
          'direct sub-directories of the URI of the system generated artifact.'
      ):
        execution_publish_utils.publish_succeeded_execution(
            m, execution_id, contexts, output_dict, executor_output)

  def testPublishSuccessExecutionUpdatesCustomProperties(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      executor_output = text_format.Parse(
          """
          execution_properties {
          key: "int"
          value {
            int_value: 1
          }
          }
          execution_properties {
            key: "string"
            value {
              string_value: "string_value"
            }
          }
           """, execution_result_pb2.ExecutorOutput())
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      execution_publish_utils.publish_succeeded_execution(
          m, execution_id, contexts, {}, executor_output)
      [execution] = m.store.get_executions_by_id([execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: COMPLETE
          custom_properties {
            key: "int"
            value {
              int_value: 1
            }
          }
          custom_properties {
            key: "string"
            value {
              string_value: "string_value"
            }
          }
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])

  def testPublishSuccessExecutionRecordExecutionResult(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      executor_output = text_format.Parse(
          """
        execution_result {
          code: 0
          result_message: 'info message.'
         }
      """, execution_result_pb2.ExecutorOutput())
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      execution_publish_utils.publish_failed_execution(m, contexts,
                                                       execution_id,
                                                       executor_output)
      [execution] = m.store.get_executions_by_id([execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          custom_properties {
            key: '__execution_result__'
            value {
              string_value: '{\\n  "resultMessage": "info message."\\n}'
            }
          }
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])
      # No events because there is no artifact published.
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertEmpty(events)
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])

  def testPublishSuccessExecutionDropsEmptyResult(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      executor_output = text_format.Parse(
          """
        execution_result {
          code: 0
         }
      """, execution_result_pb2.ExecutorOutput())
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      execution_publish_utils.publish_failed_execution(m, contexts,
                                                       execution_id,
                                                       executor_output)
      [execution] = m.store.get_executions_by_id([execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])

  def testPublishFailedExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      executor_output = text_format.Parse(
          """
        execution_result {
          code: 1
          result_message: 'error message.'
         }
      """, execution_result_pb2.ExecutorOutput())
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      execution_publish_utils.publish_failed_execution(m, contexts,
                                                       execution_id,
                                                       executor_output)
      [execution] = m.store.get_executions_by_id([execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          custom_properties {
            key: '__execution_result__'
            value {
              string_value: '{\\n  "resultMessage": "error message.",\\n  "code": 1\\n}'
            }
          }
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])
      # No events because there is no artifact published.
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertEmpty(events)
      # Verifies the context-execution edges are set up.
      self.assertCountEqual(
          [c.id for c in contexts],
          [c.id for c in m.store.get_contexts_by_execution(execution.id)])

  def testPublishInternalExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      execution_id = execution_publish_utils.register_execution(
          m, self._execution_type, contexts).id
      output_example = standard_artifacts.Examples()
      execution_publish_utils.publish_internal_execution(
          m,
          contexts,
          execution_id,
          output_artifacts={'examples': [output_example]})
      [execution] = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
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
          type: INTERNAL_OUTPUT
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


if __name__ == '__main__':
  tf.test.main()
