# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.pipeline_state."""

import os

import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


def _test_pipeline(pipeline_id,
                   execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
                       pipeline_pb2.Pipeline.ASYNC),
                   param=1):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
  pipeline.nodes[0].pipeline_node.parameters.parameters[
      'param'].field_value.int_value = param
  return pipeline


class NodeStateTest(test_utils.TfxTest):

  def test_node_state_update(self):
    node_state = pstate.NodeState()
    self.assertEqual(pstate.NodeState.STARTED, node_state.state)
    self.assertIsNone(node_state.status)

    status = status_lib.Status(code=status_lib.Code.CANCELLED, message='foobar')
    node_state.update(pstate.NodeState.STOPPING, status)
    self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
    self.assertEqual(status, node_state.status)

    node_state.update(pstate.NodeState.STARTING)
    self.assertEqual(pstate.NodeState.STARTING, node_state.state)
    self.assertIsNone(node_state.status)


class PipelineStateTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

  def test_new_pipeline_state(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pipeline_state = pstate.PipelineState.new(m, pipeline)

      mlmd_contexts = pstate.get_orchestrator_contexts(m)
      self.assertLen(mlmd_contexts, 1)

      mlmd_executions = m.store.get_executions_by_context(mlmd_contexts[0].id)
      self.assertLen(mlmd_executions, 1)
      with pipeline_state:
        self.assertProtoPartiallyEquals(
            mlmd_executions[0],
            pipeline_state._execution,
            ignored_fields=[
                'create_time_since_epoch', 'last_update_time_since_epoch'
            ])

      self.assertEqual(pipeline, pipeline_state.pipeline)
      self.assertEqual(
          task_lib.PipelineUid.from_pipeline(pipeline),
          pipeline_state.pipeline_uid)

  def test_load_pipeline_state(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pstate.PipelineState.new(m, pipeline)

      mlmd_contexts = pstate.get_orchestrator_contexts(m)
      self.assertLen(mlmd_contexts, 1)

      mlmd_executions = m.store.get_executions_by_context(mlmd_contexts[0].id)
      self.assertLen(mlmd_executions, 1)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        self.assertProtoPartiallyEquals(mlmd_executions[0],
                                        pipeline_state._execution)

      self.assertEqual(pipeline, pipeline_state.pipeline)
      self.assertEqual(
          task_lib.PipelineUid.from_pipeline(pipeline),
          pipeline_state.pipeline_uid)

  def test_load_from_orchestrator_context(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pstate.PipelineState.new(m, pipeline)

      mlmd_contexts = pstate.get_orchestrator_contexts(m)
      self.assertLen(mlmd_contexts, 1)

      mlmd_contexts = pstate.get_orchestrator_contexts(m)
      self.assertLen(mlmd_contexts, 1)

      mlmd_executions = m.store.get_executions_by_context(mlmd_contexts[0].id)
      self.assertLen(mlmd_executions, 1)
      with pstate.PipelineState.load_from_orchestrator_context(
          m, mlmd_contexts[0]) as pipeline_state:
        self.assertProtoPartiallyEquals(mlmd_executions[0],
                                        pipeline_state._execution)

      self.assertEqual(pipeline, pipeline_state.pipeline)
      self.assertEqual(
          task_lib.PipelineUid.from_pipeline(pipeline),
          pipeline_state.pipeline_uid)

  def test_new_pipeline_state_when_pipeline_already_exists(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pstate.PipelineState.new(m, pipeline)

      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pstate.PipelineState.new(m, pipeline)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS,
                       exception_context.exception.code)

  def test_load_pipeline_state_when_no_active_pipeline(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)

      # No such pipeline so NOT_FOUND error should be raised.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pstate.PipelineState.load(m, pipeline_uid)
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

      pipeline_state = pstate.PipelineState.new(m, pipeline)

      # No error as there's an active pipeline.
      pstate.PipelineState.load(m, pipeline_uid)

      # Inactivate the pipeline.
      with pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)

      # No active pipeline so NOT_FOUND error should be raised.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        with pstate.PipelineState.load(m, pipeline_uid):
          pass
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

  def test_pipeline_stop_initiation(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      with pstate.PipelineState.new(m, pipeline) as pipeline_state:
        self.assertIsNone(pipeline_state.stop_initiated_reason())
        status = status_lib.Status(
            code=status_lib.Code.CANCELLED, message='foo bar')
        pipeline_state.initiate_stop(status)
        self.assertEqual(status, pipeline_state.stop_initiated_reason())

      # Reload from MLMD and verify.
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        self.assertEqual(status, pipeline_state.stop_initiated_reason())

  def test_update_initiation_and_apply(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1', param=1)
      updated_pipeline = _test_pipeline('pipeline1', param=2)
      with pstate.PipelineState.new(m, pipeline) as pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        pipeline_state.initiate_update(updated_pipeline)
        self.assertTrue(pipeline_state.is_update_initiated())

      # Reload from MLMD and verify.
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())
        self.assertEqual(pipeline, pipeline_state.pipeline)
        pipeline_state.apply_pipeline_update()
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())
        self.assertEqual(updated_pipeline, pipeline_state.pipeline)

      # Update should fail if execution mode is different.
      updated_pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                    'Updating execution_mode.*not supported'):
          pipeline_state.initiate_update(updated_pipeline)

      # Update should fail if pipeline structure changed.
      updated_pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC)
      updated_pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                    'Updating execution_mode.*not supported'):
          pipeline_state.initiate_update(updated_pipeline)

  def test_initiate_node_start_stop(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      node_uid = task_lib.NodeUid(
          node_id='Trainer',
          pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline))
      with pstate.PipelineState.new(m, pipeline) as pipeline_state:
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTING)
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTING, node_state.state)

      # Reload from MLMD and verify node is started.
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTING, node_state.state)

        # Set node state to STOPPING.
        status = status_lib.Status(
            code=status_lib.Code.ABORTED, message='foo bar')
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPING, status)
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status, node_state.status)

      # Reload from MLMD and verify node is stopped.
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status, node_state.status)

        # Set node state to STARTED.
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTED)
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

      # Reload from MLMD and verify node is started.
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  def test_save_and_remove_property(self):
    property_key = 'key'
    property_value = 'value'
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      with pstate.PipelineState.new(m, pipeline) as pipeline_state:
        pipeline_state.save_property(property_key, property_value)

      mlmd_contexts = pstate.get_orchestrator_contexts(m)
      mlmd_executions = m.store.get_executions_by_context(mlmd_contexts[0].id)
      self.assertLen(mlmd_executions, 1)
      self.assertIsNotNone(
          mlmd_executions[0].custom_properties.get(property_key))
      self.assertEqual(
          mlmd_executions[0].custom_properties.get(property_key).string_value,
          property_value)

      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)) as pipeline_state:
        pipeline_state.remove_property(property_key)

      mlmd_executions = m.store.get_executions_by_context(mlmd_contexts[0].id)
      self.assertLen(mlmd_executions, 1)
      self.assertIsNone(mlmd_executions[0].custom_properties.get(property_key))

  def test_async_pipeline_views(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('pipeline1')
      with pstate.PipelineState.new(m, pipeline, {
          'foo': 1,
          'bar': 'baz'
      }) as pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)

      views = pstate.PipelineView.load_all(
          m, task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertLen(views, 1)
      self.assertProtoEquals(pipeline, views[0].pipeline)
      self.assertEqual({'foo': 1, 'bar': 'baz'}, views[0].pipeline_run_metadata)

      pstate.PipelineState.new(m, pipeline)
      views = pstate.PipelineView.load_all(
          m, task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertLen(views, 2)
      self.assertProtoEquals(pipeline, views[0].pipeline)
      self.assertProtoEquals(pipeline, views[1].pipeline)

  def test_sync_pipeline_views(self):

    def _create_sync_pipeline(pipeline_id: str, run_id: str):
      pipeline = _test_pipeline(pipeline_id, pipeline_pb2.Pipeline.SYNC)
      pipeline.runtime_spec.pipeline_run_id.runtime_parameter.name = (
          constants.PIPELINE_RUN_ID_PARAMETER_NAME)
      pipeline.runtime_spec.pipeline_run_id.runtime_parameter.type = (
          pipeline_pb2.RuntimeParameter.STRING)
      runtime_parameter_utils.substitute_runtime_parameter(
          pipeline, {
              constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id,
          })
      return pipeline

    with self._mlmd_connection as m:
      pipeline = _create_sync_pipeline('pipeline', '001')
      with pstate.PipelineState.new(m, pipeline, {
          'foo': 1,
          'bar': 'baz'
      }) as pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)
        pipeline_state.save_property(pstate._PIPELINE_STATUS_MSG, 'msg')

      views = pstate.PipelineView.load_all(
          m, task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertLen(views, 1)
      self.assertEqual(views[0].pipeline_run_id, '001')
      self.assertEqual(views[0].pipeline_status_message, 'msg')
      self.assertEqual({'foo': 1, 'bar': 'baz'}, views[0].pipeline_run_metadata)
      self.assertProtoEquals(pipeline, views[0].pipeline)

      pipeline2 = _create_sync_pipeline('pipeline', '002')
      pstate.PipelineState.new(m, pipeline2)

      views = pstate.PipelineView.load_all(
          m, task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertLen(views, 2)
      views_dict = {view.pipeline_run_id: view for view in views}
      self.assertCountEqual(['001', '002'], views_dict.keys())
      self.assertProtoEquals(pipeline, views_dict['001'].pipeline)
      self.assertProtoEquals(pipeline2, views_dict['002'].pipeline)
      views_status_messages = {view.pipeline_status_message for view in views}
      self.assertEqual(views_status_messages, {'', 'msg'})

      view1 = pstate.PipelineView.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline), '001')
      view2 = pstate.PipelineView.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline), '002')
      latest_view = pstate.PipelineView.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline))
      self.assertProtoEquals(pipeline, view1.pipeline)
      self.assertProtoEquals(pipeline2, view2.pipeline)
      self.assertProtoEquals(pipeline2, latest_view.pipeline)


if __name__ == '__main__':
  tf.test.main()
