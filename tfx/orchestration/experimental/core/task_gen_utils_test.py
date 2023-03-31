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
"""Tests for tfx.orchestration.experimental.core.task_gen_utils."""

import os
import time
import uuid

from absl.testing import parameterized
import tensorflow as tf
from tfx import types
from tfx import version
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.orchestration.experimental.core.testing import test_dynamic_exec_properties_pipeline
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import status as status_lib
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2

State = metadata_store_pb2.Execution.State


class TaskGenUtilsTest(parameterized.TestCase, tu.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
    self._pipeline_root = pipeline_root

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._mlmd_connection_manager = mlmd_cm.MLMDConnectionManager.sqlite(
        metadata_path)
    self.enter_context(self._mlmd_connection_manager)
    self._mlmd_connection = self._mlmd_connection_manager.primary_mlmd_handle

    # Sets up the pipeline.
    pipeline = test_async_pipeline.create_pipeline()
    self._pipeline = pipeline
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)
    self._pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_0')

    # Extracts components.
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

  def _set_pipeline_context(self, pipeline, key, name):
    for node in [n.pipeline_node for n in pipeline.nodes]:
      for c in node.contexts.contexts:
        if c.type.name == key:
          c.name.field_value.string_value = name
          break

  def test_get_executions(self):
    with self._mlmd_connection as m:
      for node in [n.pipeline_node for n in self._pipeline.nodes]:
        self.assertEmpty(task_gen_utils.get_executions(m, node))

    # Create executions for the same nodes under different pipeline contexts.
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline1')
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 2, 1)
    otu.fake_component_output(self._mlmd_connection, self._transform)
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline2')
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 2, 1)
    otu.fake_component_output(self._mlmd_connection, self._transform)

    # Get all executions across all pipeline contexts.
    with self._mlmd_connection as m:
      all_eg_execs = sorted(
          m.store.get_executions_by_type(self._example_gen.node_info.type.name),
          key=lambda e: e.id)
      all_transform_execs = sorted(
          m.store.get_executions_by_type(self._transform.node_info.type.name),
          key=lambda e: e.id)

    # Check that correct executions are returned for each node in each pipeline.
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline1')
    with self._mlmd_connection as m:
      self.assertCountEqual(all_eg_execs[0:2],
                            task_gen_utils.get_executions(m, self._example_gen))
      self.assertCountEqual(all_transform_execs[0:1],
                            task_gen_utils.get_executions(m, self._transform))
      self.assertEmpty(task_gen_utils.get_executions(m, self._trainer))
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline2')
    with self._mlmd_connection as m:
      self.assertCountEqual(all_eg_execs[2:],
                            task_gen_utils.get_executions(m, self._example_gen))
      self.assertCountEqual(all_transform_execs[1:],
                            task_gen_utils.get_executions(m, self._transform))
      self.assertEmpty(task_gen_utils.get_executions(m, self._trainer))

  def test_get_executions_only_active(self):
    with self._mlmd_connection as m:
      for node in [n.pipeline_node for n in self._pipeline.nodes]:
        self.assertEmpty(task_gen_utils.get_executions(m, node))

    # Create executions for the same nodes under different pipeline contexts.
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline1')
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.NEW)
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.RUNNING)
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.COMPLETE)
    otu.fake_component_output(self._mlmd_connection, self._transform)
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline2')
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.NEW)
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.RUNNING)
    otu.fake_example_gen_execution_with_state(self._mlmd_connection,
                                              self._example_gen, State.COMPLETE)
    otu.fake_component_output(self._mlmd_connection, self._transform)

    # Get all ExampleGen executions across all pipeline contexts.
    with self._mlmd_connection as m:
      all_eg_execs = sorted(
          m.store.get_executions_by_type(self._example_gen.node_info.type.name),
          key=lambda e: e.id)
      active_eg_execs = [
          execution for execution in all_eg_execs
          if execution.last_known_state == State.RUNNING or
          execution.last_known_state == State.NEW
      ]

    # Check that correct executions are returned for each node in each pipeline.
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline1')
    with self._mlmd_connection as m:
      self.assertCountEqual(
          active_eg_execs[0:2],
          task_gen_utils.get_executions(m, self._example_gen, only_active=True))
      self.assertEmpty(
          task_gen_utils.get_executions(m, self._transform, only_active=True))
      self.assertEmpty(
          task_gen_utils.get_executions(m, self._trainer, only_active=True))
    self._set_pipeline_context(self._pipeline, 'pipeline', 'my_pipeline2')
    with self._mlmd_connection as m:
      self.assertCountEqual(
          active_eg_execs[2:],
          task_gen_utils.get_executions(m, self._example_gen, only_active=True))
      self.assertEmpty(
          task_gen_utils.get_executions(m, self._transform, only_active=True))
      self.assertEmpty(
          task_gen_utils.get_executions(m, self._trainer, only_active=True))

  def test_generate_task_from_active_execution(self):
    with self._mlmd_connection as m:
      # No tasks generated without running execution.
      executions = task_gen_utils.get_executions(m, self._trainer)
      self.assertIsNone(
          task_gen_utils.generate_cancel_task_from_running_execution(
              m, self._pipeline, self._trainer, executions,
              task_lib.NodeCancelType.CANCEL_EXEC))

    # Next, ensure an active execution for trainer.
    exec_properties = {'int_arg': 24, 'list_bool_arg': [True, False]}
    otu.fake_component_output(
        self._mlmd_connection, self._trainer, exec_properties=exec_properties)
    with self._mlmd_connection as m:
      execution = m.store.get_executions()[0]
      execution.last_known_state = metadata_store_pb2.Execution.RUNNING
      m.store.put_executions([execution])

      # Check that task can be generated.
      executions = task_gen_utils.get_executions(m, self._trainer)
      task = task_gen_utils.generate_cancel_task_from_running_execution(
          m, self._pipeline, self._trainer, executions,
          task_lib.NodeCancelType.CANCEL_EXEC)
      self.assertEqual(execution.id, task.execution_id)
      self.assertEqual(exec_properties, task.exec_properties)

      # Mark execution complete. No tasks should be generated.
      execution = m.store.get_executions()[0]
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution])
      executions = task_gen_utils.get_executions(m, self._trainer)
      self.assertIsNone(
          task_gen_utils.generate_cancel_task_from_running_execution(
              m, self._pipeline, self._trainer, executions,
              task_lib.NodeCancelType.CANCEL_EXEC))

  def test_generate_resolved_info(self):
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 2, 1)
    with self._mlmd_connection_manager as mlmd_connection_manager:
      resolved_info = task_gen_utils.generate_resolved_info(
          mlmd_connection_manager, self._transform)
      self.assertCountEqual(['my_pipeline', 'my_pipeline.my_transform'],
                            [c.name for c in resolved_info.contexts])
      self.assertLen(
          resolved_info.input_and_params[0].input_artifacts['examples'], 1)
      self.assertProtoPartiallyEquals(
          f"""
          id: 1
          uri: "my_examples_uri"
          custom_properties {{
            key: "span"
            value {{
              int_value: 2
            }}
          }}
          custom_properties {{
            key: '{artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY}'
            value {{string_value: "{version.__version__}"}}
          }}
          custom_properties {{
            key: "version"
            value {{
              int_value: 1
            }}
          }}
          state: LIVE""",
          resolved_info.input_and_params[0]
          .input_artifacts['examples'][0]
          .mlmd_artifact,
          ignored_fields=[
              'type_id',
              'type',
              'create_time_since_epoch',
              'last_update_time_since_epoch',
          ],
      )

  def test_generate_resolved_info_with_dynamic_exec_prop(self):
    dynamic_exec_properties_pipeline = (
        test_dynamic_exec_properties_pipeline.create_pipeline())
    self._dynamic_exec_properties_pipeline = dynamic_exec_properties_pipeline
    self._pipeline_runtime_spec = dynamic_exec_properties_pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        self._pipeline_root)
    self._pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_dynamic_prop')

    self._upstream_node = (
        dynamic_exec_properties_pipeline.nodes[0].pipeline_node)
    self._dynamic_exec_properties_node = (
        dynamic_exec_properties_pipeline.nodes[1].pipeline_node)

    self._set_pipeline_context(self._dynamic_exec_properties_pipeline,
                               'pipeline_run', 'test_run_dynamic_prop')
    for input_spec in self._dynamic_exec_properties_node.inputs.inputs.values():
      for channel in input_spec.channels:
        for context_query in channel.context_queries:
          if context_query.type.name == 'pipeline_run':
            context_query.name.field_value.string_value = 'test_run_dynamic_prop'

    otu.fake_upstream_node_run(self._mlmd_connection, self._upstream_node,
                               self.create_tempfile().full_path)
    with self._mlmd_connection_manager as mlmd_connection_manager:
      resolved_info = task_gen_utils.generate_resolved_info(
          mlmd_connection_manager, self._dynamic_exec_properties_node)

      self.assertCountEqual([
          'my_pipeline', 'test_run_dynamic_prop',
          'my_pipeline.DownstreamComponent'
      ], [c.name for c in resolved_info.contexts])
      self.assertLen(
          resolved_info.input_and_params[0]
          .input_artifacts['_UpstreamComponent.num'], 1)
      self.assertEqual(
          otu.OUTPUT_NUM,
          resolved_info.input_and_params[0].exec_properties['input_num'])

  @parameterized.named_parameters(
      dict(
          testcase_name='per_execution_idx_latest',
          execution_info_groups=[[
              dict(external_execution_index=0),
              dict(external_execution_index=1)
          ], [dict(external_execution_index=0)
             ], [dict(external_execution_index=0)]],
          expected_returned_execution_indices=[3, 1]),
      dict(
          testcase_name='newer_timestamp',
          execution_info_groups=[[
              dict(external_execution_index=0),
              dict(external_execution_index=1)
          ], [dict(external_execution_index=0),
              dict(external_execution_index=1)]],
          expected_returned_execution_indices=[2, 3])
  )
  def test_get_latest_execution_set(self, execution_info_groups,
                                    expected_returned_execution_indices):
    execution_type = metadata_store_pb2.ExecutionType(name='my_ex_type')

    with self._mlmd_connection as m:
      # Construct execution sets.
      executions = []
      for execution_info_group in execution_info_groups:
        input_and_params = [
            task_gen_utils.InputAndParam(input_artifacts={
                'input_example': [standard_artifacts.Examples()]
            })
        ] * len(execution_info_group)
        execution_group = []
        for idx, execution_info in enumerate(execution_info_group):
          input_and_param = input_and_params[idx]
          external_execution_index = execution_info['external_execution_index']
          execution = execution_lib.prepare_execution(
              m,
              execution_type,
              metadata_store_pb2.Execution.NEW,
              input_and_param.exec_properties,
              execution_name=str(uuid.uuid4()))
          if external_execution_index is not None:
            execution.custom_properties[
                task_gen_utils
                ._EXTERNAL_EXECUTION_INDEX].int_value = external_execution_index
          execution_group.append(execution)
        executions.extend(
            execution_lib.put_executions(m, execution_group, {}, [
                input_and_param.input_artifacts
                for input_and_param in input_and_params
            ]))
        # sleep 10 ms to make sure two groups executions have different
        # `create_time_since_epoch`
        time.sleep(0.01)

      # Get expected results.
      expected_execution_set = [
          executions[i] for i in expected_returned_execution_indices
      ]

      # Call the target function and test against the expected results.
      executions = m.store.get_executions()
      self.assertLen(executions, sum([len(g) for g in execution_info_groups]))

      latest_execution_set = task_gen_utils.get_latest_executions_set(
          executions)

      for expected_execution, actual_execution in zip(expected_execution_set,
                                                      latest_execution_set):
        self.assertProtoPartiallyEquals(
            expected_execution,
            actual_execution,
            ignored_fields=[
                'type',
                'create_time_since_epoch',
                'last_update_time_since_epoch',
            ],
        )

  def test_register_executions(self):
    with self._mlmd_connection as m:
      context_type = metadata_store_pb2.ContextType(name='my_ctx_type')
      context_type_id = m.store.put_context_type(context_type)
      context_1 = metadata_store_pb2.Context(
          name='context-1', type_id=context_type_id)
      context_2 = metadata_store_pb2.Context(
          name='context-2', type_id=context_type_id)
      m.store.put_contexts([context_1, context_2])

      # Registers two executions.
      task_gen_utils.register_executions(
          m,
          execution_type=metadata_store_pb2.ExecutionType(name='my_ex_type'),
          contexts=[context_1, context_2],
          input_and_params=[
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              }),
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              })
          ])

      [context_1, context_2] = m.store.get_contexts()
      self.assertLen(m.store.get_executions(), 2)
      self.assertLen(m.store.get_executions_by_context(context_1.id), 2)
      self.assertLen(m.store.get_executions_by_context(context_2.id), 2)

  def test_get_executions_num_of_failure(self):
    failed_execution = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED)
    failed_execution.custom_properties[
        task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 1

    e1 = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED)
    e1.custom_properties[task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 0

    e2 = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED)
    e2.custom_properties[task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 1

    e3 = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.RUNNING)
    e3.custom_properties[task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 1

    e4 = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED)
    e4.custom_properties[task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 1

    e5 = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED)
    e5.custom_properties[task_gen_utils._EXTERNAL_EXECUTION_INDEX].int_value = 1

    executions = [e1, e2, e3, e4, e5]
    self.assertEqual(
        3,  # e2, e4 and e5 are failed
        task_gen_utils.get_num_of_failures_from_failed_execution(
            executions, failed_execution
        ),
    )

  def test_register_execution_from_existing_execution(self):
    with self._mlmd_connection as m:
      # Put contexts.
      context_type = metadata_store_pb2.ContextType(name='my_ctx_type')
      context_type_id = m.store.put_context_type(context_type)
      contexts = [
          metadata_store_pb2.Context(name='context-1', type_id=context_type_id),
          metadata_store_pb2.Context(name='context-2', type_id=context_type_id)
      ]
      m.store.put_contexts(contexts)

      # Put a failed execution.
      input_and_param = task_gen_utils.InputAndParam(
          input_artifacts={'input_example': [standard_artifacts.Examples()]})
      execution_type = metadata_store_pb2.ExecutionType(name='my_ex_type')
      failed_execution = execution_lib.prepare_execution(
          m,
          execution_type,
          metadata_store_pb2.Execution.FAILED,
          input_and_param.exec_properties,
          execution_name=str(uuid.uuid4()))
      failed_execution.custom_properties[
          task_gen_utils
          ._EXTERNAL_EXECUTION_INDEX].int_value = 1
      failed_execution.custom_properties['should_not_be_copied'].int_value = 1
      failed_execution = execution_lib.put_execution(
          m,
          failed_execution,
          contexts,
          input_artifacts=input_and_param.input_artifacts)

      # Register a retry execution from a failed execution.
      [retry_execution] = (
          task_gen_utils.register_executions_from_existing_executions(
              m, self._example_gen, [failed_execution]
          )
      )

      self.assertEqual(
          retry_execution.last_known_state, metadata_store_pb2.Execution.NEW
      )
      self.assertEqual(
          retry_execution.custom_properties[
              task_gen_utils._EXTERNAL_EXECUTION_INDEX],
          failed_execution.custom_properties[
              task_gen_utils._EXTERNAL_EXECUTION_INDEX])
      self.assertIsNone(
          retry_execution.custom_properties.get('should_not_be_copied'))
      # Check all input artifacts are the same.
      retry_execution_inputs = execution_lib.get_input_artifacts(
          m, retry_execution.id)
      failed_execution_inputs = execution_lib.get_input_artifacts(
          m, failed_execution.id)
      self.assertEqual(retry_execution_inputs.keys(),
                       failed_execution_inputs.keys())
      for key in retry_execution_inputs:
        retry_execution_artifacts_ids = sorted(
            [a.id for a in retry_execution_inputs[key]])
        failed_execution_artifacts_ids = sorted(
            [a.id for a in failed_execution_inputs[key]])
        self.assertEqual(retry_execution_artifacts_ids,
                         failed_execution_artifacts_ids)

      [context_1, context_2] = m.store.get_contexts()
      self.assertLen(m.store.get_executions_by_context(context_1.id), 2)
      self.assertLen(m.store.get_executions_by_context(context_2.id), 2)

  def test_update_external_artifact_type(self):
    artifact_type = metadata_store_pb2.ArtifactType(name='my_type')
    artifact_pb = metadata_store_pb2.Artifact(type_id=artifact_type.id)
    artifact = types.artifact.Artifact(artifact_type)
    artifact.set_mlmd_artifact(artifact_pb)
    artifact.is_external = True

    with self._mlmd_connection as m:
      task_gen_utils.update_external_artifact_type(m, [artifact])

      artifact_types_in_local = m.store.get_artifact_types()
      self.assertLen(artifact_types_in_local, 1)
      self.assertEqual('my_type', artifact_types_in_local[0].name)
      # artifact should have the new type id.
      self.assertEqual(artifact_types_in_local[0].id, artifact_pb.type_id)

  def test_get_unprocessed_inputs(self):
    with self._mlmd_connection as m:
      # Prepare context.
      context_type = metadata_store_pb2.ContextType(name='ctx_type')
      context_type_id = m.store.put_context_type(context_type)
      context = metadata_store_pb2.Context(name='ctx', type_id=context_type_id)
      m.store.put_contexts([context])

      # Prepare artifact.
      artifact_type = metadata_store_pb2.ArtifactType(name='a_type')
      artifact_type.id = m.store.put_artifact_type(artifact_type)
      artifact_pb_1 = metadata_store_pb2.Artifact(type_id=artifact_type.id)
      artifact_pb_1.id = m.store.put_artifacts([artifact_pb_1])[0]
      artifact_pb_2 = metadata_store_pb2.Artifact(type_id=artifact_type.id)
      artifact_pb_2.id = m.store.put_artifacts([artifact_pb_2])[0]
      [artifact_1, artifact_2] = artifact_utils.deserialize_artifacts(
          artifact_type, [artifact_pb_1, artifact_pb_2]
      )

      with self.subTest(name='NoInput'):
        # There is no input.
        resolved_info = task_gen_utils.ResolvedInfo(
            contexts=[context], input_and_params=[]
        )
        unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
            m, [], resolved_info, self._transform
        )
        self.assertEmpty(unprocessed_inputs)

      with self.subTest(name='OneUnprocessedInput'):
        # There is 1 unprocessed_input
        input_and_param = task_gen_utils.InputAndParam(
            input_artifacts={'examples': [artifact_1, artifact_2]}
        )
        resolved_info = task_gen_utils.ResolvedInfo(
            contexts=[context],
            input_and_params=[input_and_param],
        )
        unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
            m, [], resolved_info, self._transform
        )
        self.assertLen(unprocessed_inputs, 1)
        self.assertEqual(unprocessed_inputs[0], input_and_param)

      # Simulate that artifact_1 and artifact_2 are processed.
      execution = execution_lib.prepare_execution(
          m,
          execution_type=metadata_store_pb2.ExecutionType(name='my_ex_type'),
          state=metadata_store_pb2.Execution.COMPLETE,
      )
      execution = execution_lib.put_execution(
          m,
          execution,
          [context],
          input_artifacts={'examples': [artifact_1, artifact_2]},
      )

      with self.subTest(name='ResolvedArtifactsMatchProcessedArtifacts'):
        input_and_param = task_gen_utils.InputAndParam(
            input_artifacts={'examples': [artifact_1, artifact_2]}
        )
        resolved_info = task_gen_utils.ResolvedInfo(
            contexts=[context],
            input_and_params=[input_and_param],
        )
        unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
            m, [execution], resolved_info, self._transform
        )
        self.assertEmpty(unprocessed_inputs)

      with self.subTest(name='ResolvedArtifactsNotMatchProcessedArtifacts'):
        input_and_param = task_gen_utils.InputAndParam(
            input_artifacts={'key1': [artifact_1], 'key2': [artifact_2]}
        )
        resolved_info = task_gen_utils.ResolvedInfo(
            contexts=[context],
            input_and_params=[input_and_param],
        )
        unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
            m, [execution], resolved_info, self._transform
        )
        self.assertLen(unprocessed_inputs, 1)
        self.assertEqual(unprocessed_inputs[0], input_and_param)

  def test_get_unprocessed_inputs_no_trigger(self):
    # Set the example_gen to transform node as NO_TRIGGER.
    input_trigger = (
        self._transform.execution_options.async_trigger.input_triggers[
            'examples'
        ]
    )
    input_trigger.no_trigger = True

    # ExampleGen generates the first output.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)
    resolved_info = task_gen_utils.generate_resolved_info(
        self._mlmd_connection_manager, self._transform
    )
    unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
        self._mlmd_connection,
        [],
        resolved_info,
        self._transform,
    )

    # Should return one unprocessed input, and trigger transform once.
    self.assertLen(unprocessed_inputs, 1)

  def test_interpret_status_from_failed_execution(self):
    execution = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.COMPLETE
    )
    with self.assertRaisesRegex(
        ValueError, 'Must be called.*last_known_state = FAILED.'
    ):
      task_gen_utils.interpret_status_from_failed_execution(execution)

    execution = metadata_store_pb2.Execution(
        last_known_state=metadata_store_pb2.Execution.FAILED
    )
    self.assertEqual(
        status_lib.Status(code=status_lib.Code.UNKNOWN),
        task_gen_utils.interpret_status_from_failed_execution(execution),
    )

    # Status is created using special custom properties if they exist.
    execution.custom_properties[
        constants.EXECUTION_ERROR_MSG_KEY
    ].string_value = 'permission denied'
    self.assertEqual(
        status_lib.Status(
            code=status_lib.Code.UNKNOWN, message='permission denied'
        ),
        task_gen_utils.interpret_status_from_failed_execution(execution),
    )
    execution.custom_properties[
        constants.EXECUTION_ERROR_CODE_KEY
    ].int_value = status_lib.Code.PERMISSION_DENIED
    self.assertEqual(
        status_lib.Status(
            code=status_lib.Code.PERMISSION_DENIED, message='permission denied'
        ),
        task_gen_utils.interpret_status_from_failed_execution(execution),
    )

    # ExecutionResult, if available, has the higher precedence in determining
    # Status as that indicates the most proximate cause.
    execution_result = execution_result_pb2.ExecutionResult(
        code=status_lib.Code.DEADLINE_EXCEEDED,
        result_message='deadline exceeded',
    )
    execution_lib.set_execution_result(execution_result, execution)
    self.assertEqual(
        status_lib.Status(
            code=status_lib.Code.DEADLINE_EXCEEDED, message='deadline exceeded'
        ),
        task_gen_utils.interpret_status_from_failed_execution(execution),
    )


if __name__ == '__main__':
  tf.test.main()
