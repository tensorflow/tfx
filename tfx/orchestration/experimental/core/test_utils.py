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
"""Test utilities."""

import os
import uuid

from absl.testing.absltest import mock
from tfx import types
from tfx.orchestration.experimental.core import async_pipeline_task_gen as asptg
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2


def fake_example_gen_run(mlmd_connection, example_gen, span, version):
  """Writes fake example_gen output and successful execution to MLMD."""
  with mlmd_connection as m:
    output_example = types.Artifact(
        example_gen.outputs.outputs['output_examples'].artifact_spec.type)
    output_example.set_int_custom_property('span', span)
    output_example.set_int_custom_property('version', version)
    output_example.uri = 'my_examples_uri'
    contexts = context_lib.prepare_contexts(m, example_gen.contexts)
    execution = execution_publish_utils.register_execution(
        m, example_gen.node_info.type, contexts)
    execution_publish_utils.publish_succeeded_execution(
        m, execution.id, contexts, {
            'output_examples': [output_example],
        })


def fake_component_output(mlmd_connection,
                          component,
                          execution=None,
                          active=False):
  """Writes fake component output and execution to MLMD."""
  with mlmd_connection as m:
    output_key, output_value = next(iter(component.outputs.outputs.items()))
    output = types.Artifact(output_value.artifact_spec.type)
    output.uri = str(uuid.uuid4())
    contexts = context_lib.prepare_contexts(m, component.contexts)
    if not execution:
      execution = execution_publish_utils.register_execution(
          m, component.node_info.type, contexts)
    if not active:
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {output_key: [output]})


def create_exec_node_task(node_uid,
                          execution=None,
                          contexts=None,
                          exec_properties=None,
                          input_artifacts=None,
                          output_artifacts=None,
                          executor_output_uri=None,
                          stateful_working_dir=None,
                          pipeline=None,
                          is_cancelled=False) -> task_lib.ExecNodeTask:
  """Creates an `ExecNodeTask` for testing."""
  return task_lib.ExecNodeTask(
      node_uid=node_uid,
      execution=execution or mock.Mock(),
      contexts=contexts or [],
      exec_properties=exec_properties or {},
      input_artifacts=input_artifacts or {},
      output_artifacts=output_artifacts or {},
      executor_output_uri=executor_output_uri or '',
      stateful_working_dir=stateful_working_dir or '',
      pipeline=pipeline or mock.Mock(),
      is_cancelled=is_cancelled)


def create_node_uid(pipeline_id, node_id):
  """Creates node uid."""
  return task_lib.NodeUid(
      pipeline_uid=task_lib.PipelineUid(pipeline_id=pipeline_id),
      node_id=node_id)


def run_generator_and_test(test_case,
                           mlmd_connection,
                           generator_class,
                           pipeline,
                           task_queue,
                           use_task_queue,
                           service_job_manager,
                           num_initial_executions,
                           num_tasks_generated,
                           num_new_executions,
                           num_active_executions,
                           ignore_node_ids=None):
  """Runs generator.generate() and tests the effects."""
  with mlmd_connection as m:
    executions = m.store.get_executions()
    test_case.assertLen(
        executions, num_initial_executions,
        f'Expected {num_initial_executions} execution(s) in MLMD.')
    pipeline_state = pstate.PipelineState.new(m, pipeline)
    generator_params = dict(
        mlmd_handle=m,
        pipeline_state=pipeline_state,
        is_task_id_tracked_fn=task_queue.contains_task_id,
        service_job_manager=service_job_manager)
    if generator_class == asptg.AsyncPipelineTaskGenerator:
      generator_params['ignore_node_ids'] = ignore_node_ids
    task_gen = generator_class(**generator_params)
    tasks = task_gen.generate()
    test_case.assertLen(
        tasks, num_tasks_generated,
        f'Expected {num_tasks_generated} task(s) to be generated.')
    executions = m.store.get_executions()
    num_total_executions = num_initial_executions + num_new_executions
    test_case.assertLen(
        executions, num_total_executions,
        f'Expected {num_total_executions} execution(s) in MLMD.')
    active_executions = [
        e for e in executions if execution_lib.is_execution_active(e)
    ]
    test_case.assertLen(
        active_executions, num_active_executions,
        f'Expected {num_active_executions} active execution(s) in MLMD.')
    if use_task_queue:
      for task in tasks:
        if task_lib.is_exec_node_task(task):
          task_queue.enqueue(task)
    return tasks, active_executions


def verify_exec_node_task(test_case, pipeline, node, execution_id, task):
  """Verifies that generated ExecNodeTask has the expected properties for the node."""
  test_case.assertEqual(
      task_lib.NodeUid.from_pipeline_node(pipeline, node), task.node_uid)
  test_case.assertEqual(execution_id, task.execution.id)
  expected_context_names = ['my_pipeline', node.node_info.id]
  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
    expected_context_names.append(
        pipeline.runtime_spec.pipeline_run_id.field_value.string_value)
  expected_input_artifacts_keys = list(iter(node.inputs.inputs.keys()))
  expected_output_artifacts_keys = list(iter(node.outputs.outputs.keys()))
  output_artifact_uri = os.path.join(
      pipeline.runtime_spec.pipeline_root.field_value.string_value,
      node.node_info.id, expected_output_artifacts_keys[0], str(execution_id))
  test_case.assertCountEqual(expected_context_names,
                             [c.name for c in task.contexts])
  test_case.assertCountEqual(expected_input_artifacts_keys,
                             list(task.input_artifacts.keys()))
  test_case.assertCountEqual(expected_output_artifacts_keys,
                             list(task.output_artifacts.keys()))
  test_case.assertEqual(
      output_artifact_uri,
      task.output_artifacts[expected_output_artifacts_keys[0]][0].uri)
  test_case.assertEqual(
      os.path.join(pipeline.runtime_spec.pipeline_root.field_value.string_value,
                   node.node_info.id, '.system', 'executor_execution',
                   str(execution_id), 'executor_output.pb'),
      task.executor_output_uri)
  test_case.assertEqual(
      os.path.join(pipeline.runtime_spec.pipeline_root.field_value.string_value,
                   node.node_info.id, '.system', 'stateful_working_dir',
                   str(execution_id)), task.stateful_working_dir)
