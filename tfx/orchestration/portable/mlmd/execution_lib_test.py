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
from typing import Sequence

from absl.testing import parameterized
import tensorflow as tf
from tfx import types
from tfx import version
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils
from tfx.utils import typing_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


_DEFAULT_ARTIFACT_TYPE = standard_artifacts.Examples


def _create_tfx_artifact(uri: str) -> types.Artifact:
  tfx_artifact = _DEFAULT_ARTIFACT_TYPE()
  tfx_artifact.uri = uri
  return tfx_artifact


def _write_tfx_artifacts(
    mlmd_handle: metadata.Metadata,
    tfx_artifacts: Sequence[types.Artifact]) -> Sequence[int]:
  """Writes TFX artifacts to MLMD and updates their IDs in-place."""
  artifact_type_id = mlmd_handle.store.put_artifact_type(
      artifact_type=metadata_store_pb2.ArtifactType(
          name=_DEFAULT_ARTIFACT_TYPE.TYPE_NAME))
  for tfx_artifact in tfx_artifacts:
    tfx_artifact.type_id = artifact_type_id

  created_artifact_ids = mlmd_handle.store.put_artifacts(
      [tfx_artifact.mlmd_artifact for tfx_artifact in tfx_artifacts])
  for idx, artifact_id in enumerate(created_artifact_ids):
    tfx_artifacts[idx].id = artifact_id
  return created_artifact_ids


class ExecutionLibTest(test_case_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    mlmd_connection = metadata.Metadata(connection_config=connection_config)
    self._mlmd_handle = self.enter_context(mlmd_connection)

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
        self._mlmd_handle,
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
        self._mlmd_handle,
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
    example = standard_artifacts.Examples()
    example.uri = 'example'
    example.id = 1

    expected_artifact = metadata_store_pb2.Artifact()
    text_format.Parse("""
        id: 1
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

    result = execution_lib._create_artifact_and_event_pairs(
        self._mlmd_handle, {
            'example': [example],
        }, metadata_store_pb2.Event.INPUT)
    self.assertLen(result, 1)
    result[0][0].ClearField('type_id')
    self.assertCountEqual([(expected_artifact, expected_event)], result)

  def testPutExecutionGraph(self):
    # Prepares an input artifact. The artifact should be registered in MLMD
    # before the put_execution call.
    input_example = standard_artifacts.Examples()
    input_example.uri = 'example'
    input_example.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example.artifact_type).id
    [input_example.id
    ] = self._mlmd_handle.store.put_artifacts([input_example.mlmd_artifact])
    # Prepares an output artifact.
    output_model = standard_artifacts.Model()
    output_model.uri = 'model'
    execution = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        exec_properties={
            'p1': 1,
            'p2': '2'
        },
        state=metadata_store_pb2.Execution.COMPLETE)
    contexts = self._generate_contexts(self._mlmd_handle)
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution,
        contexts,
        input_artifacts={
            'example': [input_example],
            'another_example': [input_example]
        },
        output_artifacts={'model': [output_model]})

    self.assertProtoPartiallyEquals(
        output_model.mlmd_artifact,
        self._mlmd_handle.store.get_artifacts_by_id([output_model.id])[0],
        ignored_fields=[
            'type',
            'create_time_since_epoch',
            'last_update_time_since_epoch',
        ],
    )
    self.assertEqual(
        output_model.get_string_custom_property(
            artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY),
        version.__version__)
    # Verifies edges between artifacts and execution.
    [input_event
    ] = self._mlmd_handle.store.get_events_by_artifact_ids([input_example.id])
    self.assertEqual(input_event.execution_id, execution.id)
    self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
    self.assertLen(input_event.path.steps, 4)
    [output_event
    ] = self._mlmd_handle.store.get_events_by_artifact_ids([output_model.id])
    self.assertEqual(output_event.execution_id, execution.id)
    self.assertEqual(output_event.type, metadata_store_pb2.Event.OUTPUT)
    # Verifies edges connecting contexts and {artifacts, execution}.
    context_ids = [context.id for context in contexts]
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            input_example.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            output_model.id)
    ], context_ids)
    self.assertCountEqual([
        c.id
        for c in self._mlmd_handle.store.get_contexts_by_execution(execution.id)
    ], context_ids)

  def testGetExecutionsAssociatedWithAllContexts(self):
    contexts = self._generate_contexts(self._mlmd_handle)
    self.assertLen(contexts, 2)

    # Create 2 executions and associate with one context each.
    execution1 = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        metadata_store_pb2.Execution.RUNNING)
    execution1 = execution_lib.put_execution(self._mlmd_handle, execution1,
                                             [contexts[0]])
    execution2 = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        metadata_store_pb2.Execution.COMPLETE)
    execution2 = execution_lib.put_execution(self._mlmd_handle, execution2,
                                             [contexts[1]])

    # Create another execution and associate with both contexts.
    execution3 = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        metadata_store_pb2.Execution.NEW)
    execution3 = execution_lib.put_execution(self._mlmd_handle, execution3,
                                             contexts)

    # Verify that the right executions are returned.
    with self.subTest(for_contexts=(0,)):
      executions = execution_lib.get_executions_associated_with_all_contexts(
          self._mlmd_handle, [contexts[0]])
      self.assertCountEqual([execution1.id, execution3.id],
                            [e.id for e in executions])
    with self.subTest(for_contexts=(1,)):
      executions = execution_lib.get_executions_associated_with_all_contexts(
          self._mlmd_handle, [contexts[1]])
      self.assertCountEqual([execution2.id, execution3.id],
                            [e.id for e in executions])
    with self.subTest(for_contexts=(0, 1)):
      executions = execution_lib.get_executions_associated_with_all_contexts(
          self._mlmd_handle, contexts)
      self.assertCountEqual([execution3.id], [e.id for e in executions])

  def testGetArtifactIdsForExecutionIdGroupedByEventType(self):
    # Register an input and output artifacts in MLMD.
    input_example = standard_artifacts.Examples()
    input_example.uri = 'example'
    input_example.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example.artifact_type).id
    output_model = standard_artifacts.Model()
    output_model.uri = 'model'
    output_model.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, output_model.artifact_type).id
    [input_example.id, output_model.id] = self._mlmd_handle.store.put_artifacts(
        [input_example.mlmd_artifact, output_model.mlmd_artifact])
    execution = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        exec_properties={
            'p1': 1,
            'p2': '2'
        },
        state=metadata_store_pb2.Execution.COMPLETE)
    contexts = self._generate_contexts(self._mlmd_handle)
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution,
        contexts,
        input_artifacts={'example': [input_example]},
        output_artifacts={'model': [output_model]})

    artifact_ids_by_event_type = (
        execution_lib.get_artifact_ids_by_event_type_for_execution_id(
            self._mlmd_handle, execution.id))
    self.assertDictEqual(
        {
            metadata_store_pb2.Event.INPUT: set([input_example.id]),
            metadata_store_pb2.Event.OUTPUT: set([output_model.id]),
        }, artifact_ids_by_event_type)

  def testPutExecutions(self):
    # Prepares input artifacts.
    input_example_1 = standard_artifacts.Examples()
    input_example_1.uri = 'example'
    input_example_1.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example_1.artifact_type).id
    input_example_2 = standard_artifacts.Examples()
    input_example_2.uri = 'example'
    input_example_2.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example_2.artifact_type).id
    input_example_3 = standard_artifacts.Examples()
    input_example_3.uri = 'example'
    input_example_3.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example_3.artifact_type).id
    [input_example_1.id, input_example_2.id,
     input_example_3.id] = self._mlmd_handle.store.put_artifacts([
         input_example_1.mlmd_artifact, input_example_2.mlmd_artifact,
         input_example_3.mlmd_artifact
     ])

    # Prepares output artifacts.
    output_model_1 = standard_artifacts.Model()
    output_model_1.uri = 'model'
    output_model_2 = standard_artifacts.Model()
    output_model_2.uri = 'model'

    # Prepares executions.
    execution_1 = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        state=metadata_store_pb2.Execution.COMPLETE)
    execution_2 = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        state=metadata_store_pb2.Execution.COMPLETE)

    # Prepares contexts.
    contexts = self._generate_contexts(self._mlmd_handle)

    # Run the function for test.
    [execution_1, execution_2] = execution_lib.put_executions(
        self._mlmd_handle, [execution_1, execution_2],
        contexts,
        input_artifacts_maps=[{
            'examples': [input_example_1, input_example_2]
        }, {
            'another_examples': [input_example_3]
        }],
        output_artifacts_maps=[{
            'models': [output_model_1]
        }, {
            'another_models': [output_model_2]
        }])

    # Verifies artifacts.
    all_artifacts = self._mlmd_handle.store.get_artifacts()
    self.assertLen(all_artifacts, 5)
    [output_model_1, output_model_2] = [
        artifact for artifact in all_artifacts if artifact.id not in
        [input_example_1.id, input_example_2.id, input_example_3.id]
    ]
    for actual_output_artifact in [output_model_1, output_model_2]:
      self.assertIn(artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY,
                    actual_output_artifact.custom_properties)
      self.assertEqual(
          actual_output_artifact.custom_properties[
              artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY]
          .string_value, version.__version__)

    # Verifies edges between input artifacts and execution.
    [input_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([input_example_1.id
                                                           ]))
    self.assertEqual(input_event.execution_id, execution_1.id)
    self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
    [input_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([input_example_2.id
                                                           ]))
    self.assertEqual(input_event.execution_id, execution_1.id)
    self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
    [input_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([input_example_3.id
                                                           ]))
    self.assertEqual(input_event.execution_id, execution_2.id)
    self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)

    # Verifies edges between output artifacts and execution.
    [output_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([output_model_1.id]))
    self.assertEqual(output_event.execution_id, execution_1.id)
    self.assertEqual(output_event.type, metadata_store_pb2.Event.OUTPUT)
    [output_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([output_model_2.id]))
    self.assertEqual(output_event.execution_id, execution_2.id)
    self.assertEqual(output_event.type, metadata_store_pb2.Event.OUTPUT)

    # Verifies edges connecting contexts and {artifacts, execution}.
    context_ids = [context.id for context in contexts]
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            input_example_1.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            output_model_1.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_execution(
            execution_1.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            input_example_2.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_artifact(
            output_model_2.id)
    ], context_ids)
    self.assertCountEqual([
        c.id for c in self._mlmd_handle.store.get_contexts_by_execution(
            execution_2.id)
    ], context_ids)

  def testPutExecutions_None_Input(self):
    # Prepares input artifacts.
    input_example = standard_artifacts.Examples()
    input_example.type_id = common_utils.register_type_if_not_exist(
        self._mlmd_handle, input_example.artifact_type
    ).id
    [input_example.id] = self._mlmd_handle.store.put_artifacts(
        [input_example.mlmd_artifact]
    )
    # Prepares executions.
    execution = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        state=metadata_store_pb2.Execution.COMPLETE,
    )
    # Prepares contexts.
    contexts = self._generate_contexts(self._mlmd_handle)

    # Runs the function for test, with None input
    input_and_params = task_gen_utils.InputAndParam(input_artifacts=None)
    [execution] = execution_lib.put_executions(
        self._mlmd_handle,
        [execution],
        contexts,
        input_artifacts_maps=[input_and_params.input_artifacts],
    )

    # Verifies that events should be empty.
    events = self._mlmd_handle.store.get_events_by_artifact_ids(
        [input_example.id]
    )
    self.assertEmpty(events)

  def testGetInputsAndOutputs(self):
    # Create and shuffle a few artifacts. The shuffled order should be
    # retained in the output of `execution_lib.get_input_artifacts`.
    input_artifact_keys = ('input1', 'input2', 'input3')
    expected_input_artifacts = collections.OrderedDict()
    for input_key in input_artifact_keys:
      input_examples = []
      for i in range(10):
        input_example = standard_artifacts.Examples()
        input_example.uri = f'{input_key}/example{i}'
        input_example.type_id = common_utils.register_type_if_not_exist(
            self._mlmd_handle, input_example.artifact_type).id
        input_examples.append(input_example)
      random.shuffle(input_examples)
      expected_input_artifacts[input_key] = input_examples

    output_models = []
    for i in range(8):
      output_model = standard_artifacts.Model()
      output_model.uri = f'model{i}'
      output_model.type_id = common_utils.register_type_if_not_exist(
          self._mlmd_handle, output_model.artifact_type).id
      output_models.append(output_model)
    random.shuffle(output_models)
    expected_output_artifacts = {'model': output_models}

    # Store input artifacts only. Outputs will be saved in put_execution().
    input_mlmd_artifacts = [
        a.mlmd_artifact
        for a in itertools.chain(*expected_input_artifacts.values())
    ]
    artifact_ids = self._mlmd_handle.store.put_artifacts(input_mlmd_artifacts)
    for artifact_id, mlmd_artifact in zip(artifact_ids, input_mlmd_artifacts):
      mlmd_artifact.id = artifact_id

    execution = execution_lib.prepare_execution(
        self._mlmd_handle,
        metadata_store_pb2.ExecutionType(name='my_execution_type'),
        state=metadata_store_pb2.Execution.RUNNING)
    contexts = self._generate_contexts(self._mlmd_handle)

    # Change the order of the OrderedDict to shuffle the order of input keys.
    expected_input_artifacts.move_to_end('input1')
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution,
        contexts,
        input_artifacts=expected_input_artifacts,
        output_artifacts=expected_output_artifacts)

    # Verify that the same artifacts are returned in the correct order.
    input_artifacts = execution_lib.get_input_artifacts(self._mlmd_handle,
                                                        execution.id)
    output_artifacts = execution_lib.get_output_artifacts(
        self._mlmd_handle, execution.id)
    self.assertEqual(set(input_artifact_keys), set(input_artifacts.keys()))
    for key in input_artifacts:
      self.assertEqual([ex.uri for ex in expected_input_artifacts[key]],
                       [a.uri for a in input_artifacts[key]], f'for key={key}')
    self.assertEqual({'model'}, set(output_artifacts.keys()))
    self.assertEqual([model.uri for model in output_models],
                     [a.uri for a in output_artifacts['model']])
    self.assertEqual(output_artifacts['model'][0].mlmd_artifact.type,
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
          """,
        execution,
    )
    self.assertEqual(
        execution_result_pb2.ExecutionResult(
            code=1, result_message='error message.'
        ),
        execution_lib.get_execution_result(execution),
    )

  def test_set_execution_result_clear_metadata_details_if_error(self):
    execution = metadata_store_pb2.Execution()
    execution_result = text_format.Parse(
        """
        code: 1
        result_message: 'error message.'
      """,
        execution_result_pb2.ExecutionResult(),
    )
    execution_result.metadata_details.add().type_url = 'non_existent_type_url'
    execution_lib.set_execution_result(execution_result, execution)

    self.assertProtoEquals(
        """
          custom_properties {
            key: '__execution_result__'
            value {
              string_value: '{\\n  "resultMessage": "error message.",\\n  "code": 1\\n}'
            }
          }
          """,
        execution,
    )
    self.assertEqual(
        execution_result_pb2.ExecutionResult(
            code=1, result_message='error message.'
        ),
        execution_lib.get_execution_result(execution),
    )

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

  def testRegisterOutputArtifacts(self):
    input_example = _create_tfx_artifact(uri='example')
    _ = _write_tfx_artifacts(self._mlmd_handle, [input_example])
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution_lib.prepare_execution(
            self._mlmd_handle,
            metadata_store_pb2.ExecutionType(name='my_execution_type'),
            state=metadata_store_pb2.Execution.RUNNING),
        self._generate_contexts(self._mlmd_handle),
        input_artifacts={'example': [input_example]})

    output_model = _create_tfx_artifact(uri='model')
    output_artifacts = {'model': [output_model]}
    execution_lib.register_pending_output_artifacts(self._mlmd_handle,
                                                    execution.id,
                                                    output_artifacts)

    actual_output_artifact = self._mlmd_handle.store.get_artifacts_by_id(
        [output_model.id])[0]
    self.assertProtoPartiallyEquals(
        output_model.mlmd_artifact,
        actual_output_artifact,
        ignored_fields=[
            'type',
            'create_time_since_epoch',
            'last_update_time_since_epoch',
        ],
    )
    self.assertEqual(actual_output_artifact.state,
                     metadata_store_pb2.Artifact.PENDING)
    self.assertEqual(
        actual_output_artifact.type, _DEFAULT_ARTIFACT_TYPE.TYPE_NAME
    )

    # Verifies edges between artifacts and execution.
    [input_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([input_example.id]))
    self.assertEqual(input_event.execution_id, execution.id)
    self.assertEqual(input_event.type, metadata_store_pb2.Event.INPUT)
    self.assertLen(input_event.path.steps, 2)

    self.assertTrue(output_model.mlmd_artifact.HasField('id'))
    [pending_output_event] = (
        self._mlmd_handle.store.get_events_by_artifact_ids([output_model.id]))
    self.assertEqual(pending_output_event.execution_id, execution.id)
    self.assertEqual(pending_output_event.type,
                     metadata_store_pb2.Event.PENDING_OUTPUT)
    self.assertLen(pending_output_event.path.steps, 2)

  def testRegisterOutputArtifactsOnInactiveExecutionFails(self):
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution_lib.prepare_execution(
            self._mlmd_handle,
            metadata_store_pb2.ExecutionType(name='my_execution_type'),
            state=metadata_store_pb2.Execution.COMPLETE),
        self._generate_contexts(self._mlmd_handle))

    with self.assertRaisesRegex(
        ValueError, 'Cannot register output artifacts on inactive execution'):
      execution_lib.register_pending_output_artifacts(self._mlmd_handle,
                                                      execution.id, {})

  def testRegisterOutputArtifactsTwiceWithSameArgumentsReusesExistingArtifact(
      self):
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution_lib.prepare_execution(
            self._mlmd_handle,
            metadata_store_pb2.ExecutionType(name='my_execution_type'),
            state=metadata_store_pb2.Execution.RUNNING),
        self._generate_contexts(self._mlmd_handle))

    artifact_uri = '/model/1'
    output_model_first_call = _create_tfx_artifact(artifact_uri)
    execution_lib.register_pending_output_artifacts(
        self._mlmd_handle, execution.id, {'model': [output_model_first_call]})

    # Assert the new artifact was registered in MLMD with valid IDs.
    self.assertGreater(output_model_first_call.id, 0)
    self.assertGreater(output_model_first_call.type_id, 0)

    output_model_second_call = _create_tfx_artifact(artifact_uri)
    execution_lib.register_pending_output_artifacts(
        self._mlmd_handle, execution.id, {'model': [output_model_second_call]})

    # Assert the second call reuses the type IDs from the first call.
    self.assertEqual(output_model_first_call.id, output_model_second_call.id)
    self.assertEqual(output_model_first_call.type_id,
                     output_model_second_call.type_id)
    self.assertEqual(output_model_first_call.uri, artifact_uri)
    self.assertEqual(output_model_second_call.uri, artifact_uri)

  def testRegisterOutputArtifactsTwiceWithDifferentArgumentsRaisesError(self):
    execution = execution_lib.put_execution(
        self._mlmd_handle,
        execution_lib.prepare_execution(
            self._mlmd_handle,
            metadata_store_pb2.ExecutionType(name='my_execution_type'),
            state=metadata_store_pb2.Execution.RUNNING),
        self._generate_contexts(self._mlmd_handle))

    output_model_first_call = _create_tfx_artifact('/model/1')
    execution_lib.register_pending_output_artifacts(
        self._mlmd_handle, execution.id, {'model': [output_model_first_call]})

    output_model_second_call = _create_tfx_artifact('/model/2')
    with self.assertRaisesRegex(
        ValueError, 'Pending output artifacts were already registered'):
      execution_lib.register_pending_output_artifacts(
          self._mlmd_handle, execution.id,
          {'model': [output_model_second_call]})

  @parameterized.named_parameters(
      dict(
          testcase_name='Empty maps returns true',
          left={},
          right={},
          expected_result=True,
      ),
      dict(
          testcase_name='Full maps with equivalent content returns true',
          left={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/2')],
          },
          right={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/2')],
          },
          expected_result=True,
      ),
      dict(
          testcase_name='Missing artifact returns false',
          left={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/2')],
          },
          right={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1')],
          },
          expected_result=False,
      ),
      dict(
          testcase_name='Missing key returns false',
          left={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/2')],
          },
          right={
              'foo': [_create_tfx_artifact('a/b/c')],
          },
          expected_result=False,
      ),
      dict(
          testcase_name='Different URI returns false',
          left={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/2')],
          },
          right={
              'foo': [_create_tfx_artifact('a/b/c')],
              'bar': [_create_tfx_artifact('a/1'),
                      _create_tfx_artifact('a/3')],
          },
          expected_result=False,
      ),
  )
  def test_artifact_maps_contain_same_uris(self,
                                           left: typing_utils.ArtifactMultiMap,
                                           right: typing_utils.ArtifactMultiMap,
                                           expected_result: bool):
    self.assertEqual(
        expected_result,
        execution_lib._artifact_maps_contain_same_uris(left, right))

if __name__ == '__main__':
  tf.test.main()
