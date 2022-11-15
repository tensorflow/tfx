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

import collections
import itertools
import random
from typing import Optional, Sequence, Type

import tensorflow as tf
from tfx import version
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact as artifact_lib
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


def _build_artifact(artifact_cls: Type[artifact_lib.Artifact],
                    uri: str,
                    artifact_id: Optional[int] = None,
                    state: Optional[str] = None) -> artifact_lib.Artifact:
  artifact = artifact_cls()
  artifact.uri = uri
  if artifact_id is not None:
    artifact.id = artifact_id
  if state:
    artifact.state = state
  return artifact


def _write_artifacts(mlmd_handle: metadata.Metadata,
                     artifacts: Sequence[artifact_lib.Artifact]) -> None:
  """Writes artifacts to MLMD and adds created IDs to artifacts in-place."""
  for artifact in artifacts:
    artifact.type_id = common_utils.register_type_if_not_exist(
        mlmd_handle, artifact.artifact_type).id
  mlmd_artifacts_to_create = [artifact.mlmd_artifact for artifact in artifacts]
  for idx, artifact_id in enumerate(
      mlmd_handle.store.put_artifacts(mlmd_artifacts_to_create)):
    artifacts[idx].id = artifact_id


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
              'p2': '2',
              'p3': True,
              'p4': ['24', '56']
          },
          state=metadata_store_pb2.Execution.COMPLETE)
      result.ClearField('type_id')
      self.assertProtoEquals(
          """
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
          custom_properties {
            key: 'p3'
            value {
              string_value: 'true'
            }
          }
          custom_properties {
            key: '__schema__p3__'
            value {
              string_value: '{\\n  \\"value_type\\": {\\n    \\"boolean_type\\": {}\\n  }\\n}'
            }
          }
          custom_properties {
            key: 'p4'
            value {
              string_value: '["24", "56"]'
            }
          }
          custom_properties {
            key: '__schema__p4__'
            value {
              string_value: '{\\n  \\"value_type\\": {\\n    \\"list_type\\": {}\\n  }\\n}'
            }
          }
          """, result)

  def testPrepareExecutionWithName(self):
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
          execution_type=execution_type,
          exec_properties={},
          state=metadata_store_pb2.Execution.COMPLETE,
          execution_name='test_name')
      result.ClearField('type_id')
      self.assertProtoEquals(
          """
          last_known_state: COMPLETE
          name: 'test_name'
          """, result)

  def testArtifactAndEventPairs(self):
    original_artifact1 = _build_artifact(
        standard_artifacts.Examples, uri='example/1', artifact_id=1)
    original_artifact2 = _build_artifact(
        standard_artifacts.Examples, uri='example/2', artifact_id=2)
    input_dict = {
        'example': [original_artifact1, original_artifact2],
    }
    registered_artifact_ids = {original_artifact2.id}

    expected_artifact1 = metadata_store_pb2.Artifact()
    text_format.Parse("""
        id: 1
        uri: 'example/1'""", expected_artifact1)
    expected_artifact2 = metadata_store_pb2.Artifact()
    text_format.Parse("""
        id: 2
        uri: 'example/2'""", expected_artifact2)
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
          m, input_dict, metadata_store_pb2.Event.INPUT,
          registered_artifact_ids)
      self.assertLen(result, 2)
      first_artifact_event_pair = result[0]
      second_artifact_event_pair = result[1]
      self.assertProtoEquals(expected_artifact2, first_artifact_event_pair[0])
      self.assertIsNone(first_artifact_event_pair[1])
      self.assertProtoPartiallyEquals(expected_artifact1,
                                      second_artifact_event_pair[0],
                                      ['type_id'])
      self.assertProtoEquals(expected_event, second_artifact_event_pair[1])

  def testPutExecutionGraph(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Prepares an input artifact. The artifact should be registered in MLMD
      # before the put_execution call.
      input_example = _build_artifact(
          standard_artifacts.Examples, uri='example/1')
      _write_artifacts(m, [input_example])
      input_dict = {'example1': [input_example], 'example2': [input_example]}

      # Prepares an output artifact.
      output_model = _build_artifact(
          standard_artifacts.Model,
          uri='model',
          state=artifact_lib.ArtifactState.PENDING)
      output_dict = {'model': [output_model]}

      execution = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          exec_properties={
              'p1': 1,
              'p2': '2'
          },
          state=metadata_store_pb2.Execution.RUNNING)
      contexts = self._generate_contexts(m)
      execution = execution_lib.put_execution(
          m,
          execution,
          contexts,
          input_artifacts=input_dict,
          output_artifacts=output_dict)

      actual_output_model = m.store.get_artifacts_by_id([output_model.id])[0]
      self.assertProtoPartiallyEquals(
          output_model.mlmd_artifact,
          actual_output_model,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      self.assertEqual(
          output_model.get_string_custom_property(
              artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY),
          version.__version__)
      self.assertEqual(actual_output_model.state,
                       metadata_store_pb2.Artifact.State.PENDING)

      # Verifies edges between artifacts and execution.
      [input_event] = m.store.get_events_by_artifact_ids([input_example.id])
      self.assertEqual(input_event.execution_id, execution.id)
      self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
      self.assertLen(input_event.path.steps, 4)
      [output_event] = m.store.get_events_by_artifact_ids([output_model.id])
      self.assertEqual(output_event.execution_id, execution.id)
      self.assertEqual(output_event.type, metadata_store_pb2.Event.OUTPUT)

      # Verifies edges connecting contexts and {artifacts, execution}.
      context_ids = [context.id for context in contexts]
      for artifact in [input_example, output_model]:
        self.assertCountEqual(
            [c.id for c in m.store.get_contexts_by_artifact(artifact.id)],
            context_ids)
      self.assertCountEqual(
          [c.id for c in m.store.get_contexts_by_execution(execution.id)],
          context_ids)

      # Update the execution and output artifact state and call put_executions
      # again, which should update these existing entities.
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      output_model.state = artifact_lib.ArtifactState.PUBLISHED
      output_model.id = actual_output_model.id

      _ = execution_lib.put_execution(
          m,
          execution,
          contexts,
          input_artifacts=input_dict,
          output_artifacts=output_dict)

      # Verify the actual artifact and execution entities in MLMD were updated.
      all_artifacts_by_id = {x.id: x for x in m.store.get_artifacts()}
      self.assertLen(all_artifacts_by_id, 2)
      self.assertIn(output_model.id, all_artifacts_by_id)
      self.assertEqual(metadata_store_pb2.Artifact.State.LIVE,
                       all_artifacts_by_id[output_model.id].state)

      all_executions = m.store.get_executions()
      self.assertLen(all_executions, 1)
      self.assertEqual(metadata_store_pb2.Execution.COMPLETE,
                       all_executions[0].last_known_state)

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
      input_example = _build_artifact(
          standard_artifacts.Examples, uri='example')
      output_model = _build_artifact(standard_artifacts.Model, uri='model')
      _write_artifacts(m, [input_example, output_model])
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

  def testPutExecutions(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Prepares input artifacts.
      input_example_1 = _build_artifact(
          standard_artifacts.Examples, uri='example/1')
      input_example_2 = _build_artifact(
          standard_artifacts.Examples, uri='example/2')
      input_example_3 = _build_artifact(
          standard_artifacts.Examples, uri='example/3')
      _write_artifacts(m, [input_example_1, input_example_2, input_example_3])
      input_dicts = [{
          'examples': [input_example_1, input_example_2]
      }, {
          'another_examples': [input_example_3]
      }]

      # Prepares output artifacts.
      output_model_1 = _build_artifact(
          standard_artifacts.Model,
          uri='model/1',
          state=artifact_lib.ArtifactState.PENDING)
      output_model_2 = _build_artifact(
          standard_artifacts.Model,
          uri='model/2',
          state=artifact_lib.ArtifactState.PENDING)
      output_dicts = [{
          'models': [output_model_1]
      }, {
          'another_models': [output_model_2]
      }]

      # Prepares executions.
      execution_1 = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          state=metadata_store_pb2.Execution.RUNNING)
      execution_2 = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          state=metadata_store_pb2.Execution.RUNNING)

      # Prepares contexts.
      contexts = self._generate_contexts(m)

      # Run the function for test.
      [execution_1, execution_2] = execution_lib.put_executions(
          m, [execution_1, execution_2],
          contexts,
          input_artifacts_maps=input_dicts,
          output_artifacts_maps=output_dicts)

      # Verifies artifacts.
      all_artifacts = m.store.get_artifacts()
      self.assertLen(all_artifacts, 5)
      [actual_output_model_1, actual_output_model_2] = [
          artifact for artifact in all_artifacts if artifact.id not in
          [input_example_1.id, input_example_2.id, input_example_3.id]
      ]
      for actual_output_artifact in [
          actual_output_model_1, actual_output_model_2
      ]:
        self.assertIn(artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY,
                      actual_output_artifact.custom_properties)
        self.assertEqual(
            actual_output_artifact.custom_properties[
                artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY]
            .string_value, version.__version__)
        self.assertEqual(metadata_store_pb2.Artifact.State.PENDING,
                         actual_output_artifact.state)

      # Verifies edges between input artifacts and execution.
      for input_artifact, execution in [(input_example_1, execution_1),
                                        (input_example_2, execution_1),
                                        (input_example_3, execution_2)]:
        [event] = m.store.get_events_by_artifact_ids([input_artifact.id])
        self.assertEqual(event.execution_id, execution.id)
        self.assertEqual(event.type, metadata_store_pb2.Event.INPUT)

      # Verifies edges between output artifacts and execution.
      for output_artifact, execution in [(actual_output_model_1, execution_1),
                                         (actual_output_model_2, execution_2)]:
        [event] = m.store.get_events_by_artifact_ids([output_artifact.id])
        self.assertEqual(event.execution_id, execution.id)
        self.assertEqual(event.type, metadata_store_pb2.Event.OUTPUT)

      # Verifies edges connecting contexts and {artifacts, execution}.
      context_ids = [context.id for context in contexts]
      for artifact in [
          input_example_1, input_example_2, input_example_3,
          actual_output_model_1, actual_output_model_2
      ]:
        self.assertCountEqual(
            [c.id for c in m.store.get_contexts_by_artifact(artifact.id)],
            context_ids)
      for execution in [execution_1, execution_2]:
        self.assertCountEqual(
            [c.id for c in m.store.get_contexts_by_execution(execution.id)],
            context_ids)

      # Update the execution and output artifact state and call put_executions
      # again, which should update these existing entities.
      for execution in [execution_1, execution_2]:
        execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      for artifact in [output_model_1, output_model_2]:
        artifact.state = artifact_lib.ArtifactState.PUBLISHED
      output_model_1.id = actual_output_model_1.id
      output_model_2.id = actual_output_model_2.id

      _ = execution_lib.put_executions(
          m, [execution_1, execution_2],
          contexts,
          input_artifacts_maps=input_dicts,
          output_artifacts_maps=output_dicts)

      # Verify the actual artifact and execution entities in MLMD were updated.
      all_artifacts = m.store.get_artifacts()
      self.assertLen(all_artifacts, 5)
      actual_output_artifacts = [
          artifact for artifact in all_artifacts
          if artifact.id in [output_model_1.id, output_model_2.id]
      ]
      self.assertLen(actual_output_artifacts, 2)
      for artifact in actual_output_artifacts:
        self.assertEqual(metadata_store_pb2.Artifact.State.LIVE, artifact.state)

      all_executions = m.store.get_executions()
      self.assertLen(all_executions, 2)
      for execution in all_executions:
        self.assertEqual(metadata_store_pb2.Execution.COMPLETE,
                         execution.last_known_state)

  def testGetArtifactsDict(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      # Create and shuffle a few artifacts. The shuffled order should be
      # retained in the output of `execution_lib.get_artifacts_dict`.
      input_artifact_keys = ('input1', 'input2', 'input3')
      input_artifacts_dict = collections.OrderedDict()
      for input_key in input_artifact_keys:
        input_examples = []
        for i in range(10):
          input_example = standard_artifacts.Examples()
          input_example.uri = f'{input_key}/example{i}'
          input_example.type_id = common_utils.register_type_if_not_exist(
              m, input_example.artifact_type).id
          input_examples.append(input_example)
        random.shuffle(input_examples)
        input_artifacts_dict[input_key] = input_examples

      output_models = []
      for i in range(8):
        output_model = standard_artifacts.Model()
        output_model.uri = f'model{i}'
        output_model.type_id = common_utils.register_type_if_not_exist(
            m, output_model.artifact_type).id
        output_models.append(output_model)
      random.shuffle(output_models)
      output_artifacts_dict = {'model': output_models}

      # Store input artifacts only. Outputs will be saved in put_execution().
      input_mlmd_artifacts = [
          a.mlmd_artifact
          for a in itertools.chain(*input_artifacts_dict.values())
      ]
      artifact_ids = m.store.put_artifacts(input_mlmd_artifacts)
      for artifact_id, mlmd_artifact in zip(artifact_ids, input_mlmd_artifacts):
        mlmd_artifact.id = artifact_id

      execution = execution_lib.prepare_execution(
          m,
          metadata_store_pb2.ExecutionType(name='my_execution_type'),
          state=metadata_store_pb2.Execution.RUNNING)
      contexts = self._generate_contexts(m)

      # Change the order of the OrderedDict to shuffle the order of input keys.
      input_artifacts_dict.move_to_end('input1')
      execution = execution_lib.put_execution(
          m,
          execution,
          contexts,
          input_artifacts=input_artifacts_dict,
          output_artifacts=output_artifacts_dict)

      # Verify that the same artifacts are returned in the correct order.
      artifacts_dict = execution_lib.get_artifacts_dict(
          m, execution.id, [metadata_store_pb2.Event.INPUT])
      self.assertEqual(set(input_artifact_keys), set(artifacts_dict.keys()))
      for key in artifacts_dict:
        self.assertEqual([ex.uri for ex in input_artifacts_dict[key]],
                         [a.uri for a in artifacts_dict[key]], f'for key={key}')
      artifacts_dict = execution_lib.get_artifacts_dict(
          m, execution.id, [metadata_store_pb2.Event.OUTPUT])
      self.assertEqual({'model'}, set(artifacts_dict.keys()))
      self.assertEqual([model.uri for model in output_models],
                       [a.uri for a in artifacts_dict['model']])
      self.assertEqual(artifacts_dict['model'][0].mlmd_artifact.type,
                       standard_artifacts.Model.TYPE_NAME)

  def test_set_and_get_execution_result(self):
    execution = metadata_store_pb2.Execution()
    execution_result = text_format.Parse(
        """
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

  def test_sort_executions_newest_to_oldest(self):
    executions = [
        metadata_store_pb2.Execution(create_time_since_epoch=2),
        metadata_store_pb2.Execution(create_time_since_epoch=5),
        metadata_store_pb2.Execution(create_time_since_epoch=3),
        metadata_store_pb2.Execution(create_time_since_epoch=1),
        metadata_store_pb2.Execution(create_time_since_epoch=4)
    ]
    self.assertEqual([
        metadata_store_pb2.Execution(create_time_since_epoch=5),
        metadata_store_pb2.Execution(create_time_since_epoch=4),
        metadata_store_pb2.Execution(create_time_since_epoch=3),
        metadata_store_pb2.Execution(create_time_since_epoch=2),
        metadata_store_pb2.Execution(create_time_since_epoch=1)
    ], execution_lib.sort_executions_newest_to_oldest(executions))

  def test_is_internal_key(self):
    self.assertTrue(execution_lib.is_internal_key('__internal_key__'))
    self.assertFalse(execution_lib.is_internal_key('public_key'))


if __name__ == '__main__':
  tf.test.main()
