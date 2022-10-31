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

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.orchestration.experimental.core.testing import test_dynamic_exec_properties_pipeline
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2


class TaskGenUtilsTest(tu.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
    self._pipeline_root = pipeline_root

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

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

  def test_is_latest_execution_successful(self):
    executions = []
    self.assertFalse(task_gen_utils.is_latest_execution_successful(executions))

    # A successful execution.
    executions.append(
        metadata_store_pb2.Execution(
            id=1,
            type_id=2,
            create_time_since_epoch=10,
            last_known_state=metadata_store_pb2.Execution.COMPLETE))
    self.assertTrue(task_gen_utils.is_latest_execution_successful(executions))

    # An older failed execution should not matter.
    executions.append(
        metadata_store_pb2.Execution(
            id=2,
            type_id=2,
            create_time_since_epoch=5,
            last_known_state=metadata_store_pb2.Execution.FAILED))
    self.assertTrue(task_gen_utils.is_latest_execution_successful(executions))

    # A newer failed execution returns False.
    executions.append(
        metadata_store_pb2.Execution(
            id=3,
            type_id=2,
            create_time_since_epoch=15,
            last_known_state=metadata_store_pb2.Execution.FAILED))
    self.assertFalse(task_gen_utils.is_latest_execution_successful(executions))

    # Finally, a newer successful execution returns True.
    executions.append(
        metadata_store_pb2.Execution(
            id=4,
            type_id=2,
            create_time_since_epoch=20,
            last_known_state=metadata_store_pb2.Execution.COMPLETE))
    self.assertTrue(task_gen_utils.is_latest_execution_successful(executions))

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
    with self._mlmd_connection as m:
      resolved_info = task_gen_utils.generate_resolved_info(m, self._transform)
      self.assertCountEqual(['my_pipeline', 'my_pipeline.my_transform'],
                            [c.name for c in resolved_info.contexts])
      self.assertLen(
          resolved_info.input_and_params[0].input_artifacts['examples'], 1)
      self.assertProtoPartiallyEquals(
          """
          id: 1
          uri: "my_examples_uri"
          custom_properties {
            key: "span"
            value {
              int_value: 2
            }
          }
          custom_properties {
            key: "version"
            value {
              int_value: 1
            }
          }
          state: LIVE""",
          resolved_info.input_and_params[0].input_artifacts['examples']
          [0].mlmd_artifact,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

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
    with self._mlmd_connection as m:
      resolved_info = task_gen_utils.generate_resolved_info(
          m, self._dynamic_exec_properties_node)

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

  def test_get_latest_successful_execution(self):
    otu.fake_component_output(self._mlmd_connection, self._transform)
    otu.fake_component_output(self._mlmd_connection, self._transform)
    otu.fake_component_output(self._mlmd_connection, self._transform)
    with self._mlmd_connection as m:
      execs = sorted(m.store.get_executions(), key=lambda e: e.id)
      execs[2].last_known_state = metadata_store_pb2.Execution.FAILED
      m.store.put_executions([execs[2]])
      execs = sorted(
          task_gen_utils.get_executions(m, self._transform), key=lambda e: e.id)
      self.assertEqual(execs[1],
                       task_gen_utils.get_latest_successful_execution(execs))

  def test_get_latest_activate_execution_set(self):
    with self._mlmd_connection as m:
      # Registers two sets of executions.
      task_gen_utils.register_executions(
          m,
          metadata_store_pb2.ExecutionType(name='my_ex_type'), {},
          input_and_params=[
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              }),
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              })
          ])
      newer_execution_set = task_gen_utils.register_executions(
          m,
          metadata_store_pb2.ExecutionType(name='my_ex_type'), {},
          input_and_params=[
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              }),
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              })
          ])

      executions = m.store.get_executions()
      self.assertLen(executions, 4)

      latest_execution_set = task_gen_utils.get_latest_executions_set(
          executions)
      self.assertLen(latest_execution_set, 2)

      newer_execution_set.sort(key=lambda e: e.id)
      latest_execution_set.sort(key=lambda e: e.id)
      self.assertProtoPartiallyEquals(
          newer_execution_set[0],
          latest_execution_set[0],
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      self.assertProtoPartiallyEquals(
          newer_execution_set[1],
          latest_execution_set[1],
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def test_get_oldest_active_execution_by_index_from_a_set(self):
    with self._mlmd_connection as m:
      # Registers a set of executions.
      task_gen_utils.register_executions(
          m,
          metadata_store_pb2.ExecutionType(name='my_ex_type'), {},
          input_and_params=[
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              }),
              task_gen_utils.InputAndParam(input_artifacts={
                  'input_example': [standard_artifacts.Examples()]
              })
          ])

      # Tests the function.
      executions = m.store.get_executions()
      self.assertLen(executions, 2)
      oldest_execution = task_gen_utils.get_oldest_active_execution_by_index_from_a_set(
          executions)
      self.assertEqual(
          0,
          oldest_execution.custom_properties.get(
              task_gen_utils._EXTERNAL_EXECUTION_INDEX).int_value)

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


if __name__ == '__main__':
  tf.test.main()
