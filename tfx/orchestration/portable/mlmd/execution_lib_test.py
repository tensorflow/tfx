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

import itertools
import random

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class ExecutionLibTest(test_case_utils.TfxTest):

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
    return context_lib.prepare_contexts(metadata_handler, context_spec)

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

  def testGetExecutionsAssociatedWithAllContexts(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = self._generate_contexts(m)
      self.assertLen(contexts, 2)

      # Create 2 executions and associate with one context each.
      execution1 = execution_lib.prepare_execution(
          m, metadata_store_pb2.ExecutionType(name='my_execution_type'),
          metadata_store_pb2.Execution.RUNNING)
      execution1 = execution_lib.put_execution(m, execution1, [contexts[0]])
      execution2 = execution_lib.prepare_execution(
          m, metadata_store_pb2.ExecutionType(name='my_execution_type'),
          metadata_store_pb2.Execution.COMPLETE)
      execution2 = execution_lib.put_execution(m, execution2, [contexts[1]])

      # Create another execution and associate with both contexts.
      execution3 = execution_lib.prepare_execution(
          m, metadata_store_pb2.ExecutionType(name='my_execution_type'),
          metadata_store_pb2.Execution.NEW)
      execution3 = execution_lib.put_execution(m, execution3, contexts)

      # Verify that the right executions are returned.
      with self.subTest(for_contexts=(0,)):
        executions = execution_lib.get_executions_associated_with_all_contexts(
            m, [contexts[0]])
        self.assertCountEqual([execution1.id, execution3.id],
                              [e.id for e in executions])
      with self.subTest(for_contexts=(1,)):
        executions = execution_lib.get_executions_associated_with_all_contexts(
            m, [contexts[1]])
        self.assertCountEqual([execution2.id, execution3.id],
                              [e.id for e in executions])
      with self.subTest(for_contexts=(0, 1)):
        executions = execution_lib.get_executions_associated_with_all_contexts(
            m, contexts)
        self.assertCountEqual([execution3.id], [e.id for e in executions])

  def testGetArtifactIdsForExecutionIdGroupedByEventType(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Register an input and output artifacts in MLMD.
      input_example = standard_artifacts.Examples()
      input_example.uri = 'example'
      input_example.type_id = common_utils.register_type_if_not_exist(
          m, input_example.artifact_type).id
      output_model = standard_artifacts.Model()
      output_model.uri = 'model'
      output_model.type_id = common_utils.register_type_if_not_exist(
          m, output_model.artifact_type).id
      [input_example.id, output_model.id] = m.store.put_artifacts(
          [input_example.mlmd_artifact, output_model.mlmd_artifact])
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

      artifact_ids_by_event_type = (
          execution_lib.get_artifact_ids_by_event_type_for_execution_id(
              m, execution.id))
      self.assertDictEqual(
          {
              metadata_store_pb2.Event.INPUT: set([input_example.id]),
              metadata_store_pb2.Event.OUTPUT: set([output_model.id]),
          }, artifact_ids_by_event_type)

  def testGetArtifactsDict(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Create and shuffle a few artifacts. The shuffled order should be
      # retained in the output of `execution_lib.get_artifacts_dict`.
      input_examples = []
      for i in range(10):
        input_example = standard_artifacts.Examples()
        input_example.uri = 'example{}'.format(i)
        input_example.type_id = common_utils.register_type_if_not_exist(
            m, input_example.artifact_type).id
        input_examples.append(input_example)
      random.shuffle(input_examples)
      output_models = []
      for i in range(8):
        output_model = standard_artifacts.Model()
        output_model.uri = 'model{}'.format(i)
        output_model.type_id = common_utils.register_type_if_not_exist(
            m, output_model.artifact_type).id
        output_models.append(output_model)
      random.shuffle(output_models)
      m.store.put_artifacts([
          a.mlmd_artifact
          for a in itertools.chain(input_examples, output_models)
      ])
      execution = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          state=metadata_store_pb2.Execution.RUNNING)
      contexts = self._generate_contexts(m)
      input_artifacts_dict = {'examples': input_examples}
      output_artifacts_dict = {'model': output_models}
      execution = execution_lib.put_execution(
          m,
          execution,
          contexts,
          input_artifacts=input_artifacts_dict,
          output_artifacts=output_artifacts_dict)

      # Verify that the same artifacts are returned in the correct order.
      artifacts_dict = execution_lib.get_artifacts_dict(
          m, execution.id, metadata_store_pb2.Event.INPUT)
      self.assertCountEqual(['examples'], list(artifacts_dict.keys()))
      self.assertEqual([ex.uri for ex in input_examples],
                       [a.uri for a in artifacts_dict['examples']])
      artifacts_dict = execution_lib.get_artifacts_dict(
          m, execution.id, metadata_store_pb2.Event.OUTPUT)
      self.assertCountEqual(['model'], list(artifacts_dict.keys()))
      self.assertEqual([model.uri for model in output_models],
                       [a.uri for a in artifacts_dict['model']])

  def test_set_and_get_execution_result(self):
    execution = metadata_store_pb2.Execution()
    execution_result = text_format.Parse("""
        code: 1
        result_message: 'error message.'
      """, execution_result_pb2.ExecutionResult())
    execution_lib.set_execution_result(execution_result, execution)

    self.assertProtoEquals(
        """
          custom_properties {
            key: '__execution_result__'
            value {
              string_value: '{\\n  "resultMessage": "error message.",\\n  "code": 1\\n}'
            }
          }
          """, execution)

if __name__ == '__main__':
  tf.test.main()
