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
from typing import Optional
import uuid

from absl.testing.absltest import mock
from tfx import types
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import cache_utils
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import status as status_lib
from tfx.utils import test_case_utils
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2

OUTPUT_NUM = 33


class TfxTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    mlmd_state.clear_in_memory_state()
    pstate._PipelineIRCodec.testonly_reset()  # pylint: disable=protected-access


def fake_example_gen_run_with_handle(mlmd_handle,
                                     example_gen,
                                     span,
                                     version,
                                     is_external=False,
                                     **additional_custom_properties):
  """Writes fake example_gen output and successful execution to MLMD."""
  output_example = types.Artifact(
      example_gen.outputs.outputs['examples'].artifact_spec.type)
  output_example.set_int_custom_property('span', span)
  output_example.set_int_custom_property('version', version)
  if is_external:
    output_example.is_external = True
  for key, value in additional_custom_properties.items():
    data_types_utils.set_metadata_value(
        output_example.mlmd_artifact.custom_properties[key], value)
  output_example.uri = 'my_examples_uri'
  contexts = context_lib.prepare_contexts(mlmd_handle, example_gen.contexts)
  execution = execution_publish_utils.register_execution(
      mlmd_handle, example_gen.node_info.type, contexts)
  execution_publish_utils.publish_succeeded_execution(
      mlmd_handle, execution.id, contexts, {
          'examples': [output_example],
      })
  return execution


def fake_example_gen_run(mlmd_connection,
                         example_gen,
                         span,
                         version,
                         is_external=False):
  """Writes fake example_gen output and successful execution to MLMD."""
  with mlmd_connection as m:
    return fake_example_gen_run_with_handle(m, example_gen, span, version,
                                            is_external)


def fake_upstream_node_run(mlmd_connection: metadata.Metadata,
                           upstream_node: pipeline_pb2.PipelineNode,
                           tmp_path: str) -> metadata_store_pb2.Execution:
  """Writes fake upstream node output and successful execution to MLMD."""
  with mlmd_connection as mlmd_handle:
    num_output = standard_artifacts.Integer()
    num_output.uri = tmp_path
    num_output.value = OUTPUT_NUM
    contexts = context_lib.prepare_contexts(mlmd_handle, upstream_node.contexts)
    execution = execution_publish_utils.register_execution(
        mlmd_handle, upstream_node.node_info.type, contexts)
    execution_publish_utils.publish_succeeded_execution(mlmd_handle,
                                                        execution.id, contexts,
                                                        {
                                                            'num': [num_output],
                                                        })
    return execution


def fake_component_output_with_handle(mlmd_handle,
                                      component,
                                      execution=None,
                                      active=False,
                                      exec_properties=None):
  """Writes fake component output and execution to MLMD."""
  output_key, output_value = next(iter(component.outputs.outputs.items()))
  output = types.Artifact(output_value.artifact_spec.type)
  output.uri = str(uuid.uuid4())
  contexts = context_lib.prepare_contexts(mlmd_handle, component.contexts)
  if not execution:
    execution = execution_publish_utils.register_execution(
        mlmd_handle,
        component.node_info.type,
        contexts,
        exec_properties=exec_properties)
  if not active:
    execution_publish_utils.publish_succeeded_execution(mlmd_handle,
                                                        execution.id, contexts,
                                                        {output_key: [output]})


def fake_component_output(mlmd_connection,
                          component,
                          execution=None,
                          active=False,
                          exec_properties=None):
  """Writes fake component output and execution to MLMD."""
  with mlmd_connection as m:
    fake_component_output_with_handle(m, component, execution, active,
                                      exec_properties)


def fake_cached_execution(mlmd_connection, cache_context, component):
  """Writes cached execution; MLMD must have previous execution associated with cache_context.
  """
  with mlmd_connection as m:
    cached_outputs = cache_utils.get_cached_outputs(
        m, cache_context=cache_context)
    contexts = context_lib.prepare_contexts(m, component.contexts)
    execution = execution_publish_utils.register_execution(
        m, component.node_info.type, contexts)
    execution_publish_utils.publish_cached_execution(
        m,
        contexts=contexts,
        execution_id=execution.id,
        output_artifacts=cached_outputs)


def fake_cached_example_gen_run(mlmd_connection: metadata.Metadata,
                                example_gen: pipeline_pb2.PipelineNode):
  """Writes fake cached example gen execution to MLMD."""
  with mlmd_connection as m:
    output_example = types.Artifact(
        example_gen.outputs.outputs['examples'].artifact_spec.type)
    output_example.set_int_custom_property('span', 1)
    output_example.set_int_custom_property('version', 1)
    output_example.uri = 'my_examples_uri'
    output_example.mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    cached_outputs = {'examples': [output_example]}

    contexts = context_lib.prepare_contexts(m, example_gen.contexts)
    execution = execution_publish_utils.register_execution(
        m, example_gen.node_info.type, contexts)
    execution_publish_utils.publish_cached_execution(
        m,
        contexts=contexts,
        execution_id=execution.id,
        output_artifacts=cached_outputs)


def get_node(pipeline, node_id):
  for node in pipeline.nodes:
    node_view = node_proto_view.get_view(node)
    if node_view.node_info.id == node_id:
      return node_view
  raise ValueError(f'could not find {node_id}')


def fake_execute_node(mlmd_connection, task, artifact_custom_properties=None):
  """Simulates node execution given ExecNodeTask."""
  node = task.get_node()
  with mlmd_connection as m:
    if node.HasField('outputs'):
      output_key, output_value = next(iter(node.outputs.outputs.items()))
      output = types.Artifact(output_value.artifact_spec.type)
      if artifact_custom_properties:
        for key, val in artifact_custom_properties.items():
          if isinstance(val, int):
            output.set_int_custom_property(key, val)
          elif isinstance(val, str):
            output.set_string_custom_property(key, val)
          else:
            raise ValueError(f'unsupported type: {type(val)}')
      output.uri = str(uuid.uuid4())
      output_artifacts = {output_key: [output]}
    else:
      output_artifacts = None
    execution_publish_utils.publish_succeeded_execution(m, task.execution_id,
                                                        task.contexts,
                                                        output_artifacts)


def fake_start_node_with_handle(
    mlmd_handle, node, input_artifacts) -> metadata_store_pb2.Execution:
  """Simulates starting an execution of the given node."""
  contexts = context_lib.prepare_contexts(mlmd_handle, node.contexts)
  execution = execution_publish_utils.register_execution(
      mlmd_handle, node.node_info.type, contexts, input_artifacts)
  return execution


def fake_finish_node_with_handle(
    mlmd_handle, node, execution_id) -> Optional[typing_utils.ArtifactMultiMap]:
  """Simulates finishing an execution of the given node."""
  if node.HasField('outputs'):
    output_key, output_value = next(iter(node.outputs.outputs.items()))
    output = types.Artifact(output_value.artifact_spec.type)
    output.uri = str(uuid.uuid4())
    output_artifacts = {output_key: [output]}
  else:
    output_artifacts = None
  contexts = context_lib.prepare_contexts(mlmd_handle, node.contexts)
  return execution_publish_utils.publish_succeeded_execution(
      mlmd_handle, execution_id, contexts, output_artifacts)


def create_exec_node_task(
    node_uid,
    execution=None,
    contexts=None,
    exec_properties=None,
    input_artifacts=None,
    output_artifacts=None,
    executor_output_uri=None,
    stateful_working_dir=None,
    tmp_dir=None,
    pipeline=None,
    cancel_type: Optional[task_lib.NodeCancelType] = None
) -> task_lib.ExecNodeTask:
  """Creates an `ExecNodeTask` for testing."""
  return task_lib.ExecNodeTask(
      node_uid=node_uid,
      execution_id=execution.id if execution else 1,
      contexts=contexts or [],
      exec_properties=exec_properties or {},
      input_artifacts=input_artifacts or {},
      output_artifacts=output_artifacts or {},
      executor_output_uri=executor_output_uri or '',
      stateful_working_dir=stateful_working_dir or '',
      tmp_dir=tmp_dir or '',
      pipeline=pipeline or mock.Mock(),
      cancel_type=cancel_type)


def create_node_uid(pipeline_id, node_id, pipeline_run_id=None):
  """Creates node uid."""
  return task_lib.NodeUid(
      pipeline_uid=task_lib.PipelineUid(
          pipeline_id=pipeline_id, pipeline_run_id=pipeline_run_id),
      node_id=node_id)


def run_generator(mlmd_connection,
                  generator_class,
                  pipeline,
                  task_queue,
                  use_task_queue,
                  service_job_manager,
                  ignore_update_node_state_tasks=False,
                  fail_fast=None):
  """Generates tasks for testing."""
  with mlmd_connection as m:
    pipeline_state = get_or_create_pipeline_state(m, pipeline)
    generator_params = dict(
        mlmd_handle=m,
        is_task_id_tracked_fn=task_queue.contains_task_id,
        service_job_manager=service_job_manager)
    if fail_fast is not None:
      generator_params['fail_fast'] = fail_fast
    task_gen = generator_class(**generator_params)
    tasks = task_gen.generate(pipeline_state)
    if use_task_queue:
      for task in tasks:
        if isinstance(task, task_lib.ExecNodeTask):
          task_queue.enqueue(task)
    for task in tasks:
      if isinstance(task, task_lib.UpdateNodeStateTask):
        with pipeline_state:
          with pipeline_state.node_state_update_context(
              task.node_uid) as node_state:
            node_state.update(task.state, task.status)
  if ignore_update_node_state_tasks:
    tasks = [
        t for t in tasks if not isinstance(t, task_lib.UpdateNodeStateTask)
    ]
  return tasks


def get_non_orchestrator_executions(mlmd_handle):
  """Returns all the executions other than those of '__ORCHESTRATOR__' execution type.
  """
  executions = mlmd_handle.store.get_executions()
  result = []
  for e in executions:
    [execution_type] = mlmd_handle.store.get_execution_types_by_id([e.type_id])
    if execution_type.name != pstate._ORCHESTRATOR_RESERVED_ID:  # pylint: disable=protected-access
      result.append(e)
  return result


def get_or_create_pipeline_state(mlmd_handle, pipeline):
  """Gets or creates pipeline state for the given pipeline."""
  try:
    return pstate.PipelineState.load(
        mlmd_handle, task_lib.PipelineUid.from_pipeline(pipeline))
  except status_lib.StatusNotOkError as e:
    if e.status().code == status_lib.Code.NOT_FOUND:
      return pstate.PipelineState.new(mlmd_handle, pipeline)
    else:
      raise


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
                           expected_exec_nodes=None,
                           ignore_update_node_state_tasks=False,
                           fail_fast=None,
                           expected_context_names=None):
  """Runs generator.generate() and tests the effects."""
  if service_job_manager is None:
    service_job_manager = service_jobs.DummyServiceJobManager()
  with mlmd_connection as m:
    executions = get_non_orchestrator_executions(m)
    test_case.assertLen(
        executions, num_initial_executions,
        f'Expected {num_initial_executions} execution(s) in MLMD.')
  tasks = run_generator(
      mlmd_connection,
      generator_class,
      pipeline,
      task_queue,
      use_task_queue,
      service_job_manager,
      ignore_update_node_state_tasks=ignore_update_node_state_tasks,
      fail_fast=fail_fast)
  with mlmd_connection as m:
    test_case.assertLen(
        tasks, num_tasks_generated,
        f'Expected {num_tasks_generated} task(s) to be generated.')
    executions = get_non_orchestrator_executions(m)
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
    if expected_exec_nodes:
      for i, task in enumerate(
          t for t in tasks if isinstance(t, task_lib.ExecNodeTask)):
        _verify_exec_node_task(test_case, pipeline, expected_exec_nodes[i],
                               active_executions[i].id, task,
                               expected_context_names)
    return tasks


def _verify_exec_node_task(test_case, pipeline, node, execution_id, task,
                           expected_context_names):
  """Verifies that generated ExecNodeTask has the expected properties for the node.
  """
  if not expected_context_names:
    expected_context_names = ['my_pipeline', f'my_pipeline.{node.node_info.id}']
  test_case.assertEqual(
      task_lib.NodeUid.from_node(pipeline, node), task.node_uid)
  test_case.assertEqual(execution_id, task.execution_id)
  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
    expected_context_names.append(
        pipeline.runtime_spec.pipeline_run_id.field_value.string_value)
  expected_input_artifacts_keys = [
      key for key, value in node.inputs.inputs.items() if not value.hidden
  ]
  expected_output_artifacts_keys = list(iter(node.outputs.outputs.keys()))
  if expected_output_artifacts_keys:
    output_artifact_uri = os.path.join(
        pipeline.runtime_spec.pipeline_root.field_value.string_value,
        node.node_info.id, expected_output_artifacts_keys[0], str(execution_id))
    test_case.assertEqual(
        output_artifact_uri,
        task.output_artifacts[expected_output_artifacts_keys[0]][0].uri)
  # There may be cached context which we ignore.
  test_case.assertContainsSubset(expected_context_names,
                                 [c.name for c in task.contexts])
  test_case.assertCountEqual(expected_input_artifacts_keys,
                             list(task.input_artifacts.keys()))
  test_case.assertCountEqual(expected_output_artifacts_keys,
                             list(task.output_artifacts.keys()))
  test_case.assertEqual(
      os.path.join(pipeline.runtime_spec.pipeline_root.field_value.string_value,
                   node.node_info.id, '.system', 'executor_execution',
                   str(execution_id), 'executor_output.pb'),
      task.executor_output_uri)
  test_case.assertEqual(
      os.path.join(pipeline.runtime_spec.pipeline_root.field_value.string_value,
                   node.node_info.id, '.system', 'stateful_working_dir',
                   str(execution_id)), task.stateful_working_dir)


def concurrent_pipeline_runs_enabled_env():

  class _TestEnv(env._DefaultEnv):  # pylint: disable=protected-access

    def concurrent_pipeline_runs_enabled(self) -> bool:
      return True

  return _TestEnv()
