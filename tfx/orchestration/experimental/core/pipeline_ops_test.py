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
"""Tests for tfx.orchestration.experimental.core.pipeline_ops."""

import copy
import os
import threading
import time
from typing import Optional

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import orchestration_options
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import manual_task_scheduler
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.orchestration.experimental.core.testing import test_manual_node
from tfx.orchestration.experimental.core.testing import test_sync_pipeline
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


def _test_pipeline(
    pipeline_id: str,
    execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
        pipeline_pb2.Pipeline.ASYNC
    ),
    pipeline_run_id='run0',
    pipeline_root: Optional[str] = None,
):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  if execution_mode == pipeline_pb2.Pipeline.SYNC:
    pipeline.runtime_spec.pipeline_run_id.field_value.string_value = (
        pipeline_run_id
    )
  if pipeline_root is not None:
    pipeline.runtime_spec.pipeline_root.field_value.string_value = pipeline_root
  return pipeline


def _get_node_states_dict(
    execution: metadata_store_pb2.Execution,
) -> dict[str, pstate.NodeState]:
  return pstate._NodeStatesProxy(execution).get()


class PipelineOpsTest(test_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id(),
    )

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._mlmd_cm = mlmd_cm.MLMDConnectionManager.sqlite(metadata_path)
    self.enter_context(self._mlmd_cm)
    self._mlmd_connection = self._mlmd_cm.primary_mlmd_handle

    mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True
    )
    mock_service_job_manager.is_pure_service_node.side_effect = (
        lambda _, node_id: node_id == 'ExampleGen'
    )
    mock_service_job_manager.is_mixed_service_node.side_effect = (
        lambda _, node_id: node_id == 'Transform'
    )
    mock_service_job_manager.stop_node_services.return_value = True
    self._mock_service_job_manager = mock_service_job_manager

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
      ),
  )
  def test_initiate_pipeline_start(self, pipeline):
    with self._mlmd_connection as m:
      # Initiate a pipeline start.
      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state1:
        self.assertProtoPartiallyEquals(
            pipeline, pipeline_state1.pipeline, ignored_fields=['runtime_spec']
        )
        self.assertEqual(
            metadata_store_pb2.Execution.NEW,
            pipeline_state1.get_pipeline_execution_state(),
        )

      # Initiate another pipeline start.
      pipeline2 = _test_pipeline('pipeline2')
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline2
      ) as pipeline_state2:
        self.assertEqual(pipeline2, pipeline_state2.pipeline)
        self.assertEqual(
            metadata_store_pb2.Execution.NEW,
            pipeline_state2.get_pipeline_execution_state(),
        )

      # Error if attempted to initiate when old one is active.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.initiate_pipeline_start(m, pipeline)
      self.assertEqual(
          status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
      )

      # Fine to initiate after the previous one is inactive.
      with pipeline_state1:
        pipeline_state1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE
        )
      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state3:
        self.assertEqual(
            metadata_store_pb2.Execution.NEW,
            pipeline_state3.get_pipeline_execution_state(),
        )

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_resume_pipeline(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline(
          'test_pipeline', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run0'
      )
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['ExampleGen'])

      # Error if attempt to resume the pipeline when there is no previous run.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.resume_pipeline(
            m, pipeline, run_id='run0'
        )
      self.assertEqual(
          status_lib.Code.NOT_FOUND, exception_context.exception.code
      )

      # Initiate a pipeline start.
      pipeline_state_run0 = pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Error if attempt to resume the pipeline when the previous one is active.
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.resume_pipeline(
            m, pipeline, run_id='run0'
        )
      self.assertEqual(
          status_lib.Code.FAILED_PRECONDITION, exception_context.exception.code
      )

      with pipeline_state_run0:
        example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
        trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
        with pipeline_state_run0.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run0.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        pipeline_state_run0.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE
        )
        pipeline_state_run0.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED)
        )
      # Only Trainer is marked to run since ExampleGen succeeded in previous
      # run.
      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_base_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings, 'run0',
      )
      expected_pipeline.nodes[
          0
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.depends_on_snapshot = True
      with pipeline_ops.resume_pipeline(
          m, pipeline, run_id='run0'
      ) as pipeline_state_run1:
        self.assertEqual(expected_pipeline, pipeline_state_run1.pipeline)
        self.assertTrue(pipeline_state_run1.is_active())
        mock_snapshot.assert_called_once()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_resume_pipeline_when_concurrent_pipeline_runs_enabled(
      self, mock_snapshot
  ):
    with test_utils.concurrent_pipeline_runs_enabled_env():
      with self._mlmd_connection as m:
        pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
        pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
        node_example_gen = pipeline.nodes.add().pipeline_node
        node_example_gen.node_info.id = 'ExampleGen'
        node_example_gen.downstream_nodes.extend(['Trainer'])
        node_trainer = pipeline.nodes.add().pipeline_node
        node_trainer.node_info.id = 'Trainer'
        node_trainer.upstream_nodes.extend(['ExampleGen'])

        # Initiate a pipeline run.
        with pipeline_ops.initiate_pipeline_start(
            m, pipeline
        ) as pipeline_state:
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid(
                  task_lib.PipelineUid.from_pipeline(pipeline), 'ExampleGen'
              )
          ) as node_state:
            node_state.update(pstate.NodeState.COMPLETE)
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid(
                  task_lib.PipelineUid.from_pipeline(pipeline), 'Trainer'
              )
          ) as node_state:
            node_state.update(pstate.NodeState.FAILED)
          pipeline_state.set_pipeline_execution_state(
              metadata_store_pb2.Execution.COMPLETE
          )
          pipeline_state.initiate_stop(
              status_lib.Status(code=status_lib.Code.ABORTED)
          )

        # Initiate another pipeline run.
        pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
        with pipeline_ops.initiate_pipeline_start(
            m, pipeline
        ) as pipeline_state:
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid(
                  task_lib.PipelineUid.from_pipeline(pipeline), 'ExampleGen'
              )
          ) as node_state:
            node_state.update(pstate.NodeState.FAILED)
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid(
                  task_lib.PipelineUid.from_pipeline(pipeline), 'Trainer'
              )
          ) as node_state:
            node_state.update(pstate.NodeState.FAILED)
          pipeline_state.set_pipeline_execution_state(
              metadata_store_pb2.Execution.COMPLETE
          )
          pipeline_state.initiate_stop(
              status_lib.Status(code=status_lib.Code.ABORTED)
          )

        pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run2'

        # Error if attempt to resume the pipeline without providing run id.
        with self.assertRaises(
            status_lib.StatusNotOkError
        ) as exception_context:
          pipeline_ops.resume_pipeline(
              m,
              pipeline,
          )
        self.assertEqual(
            status_lib.Code.INVALID_ARGUMENT, exception_context.exception.code
        )

        # Success if pipeline resumed with run id.
        self.assertEqual('run0', pipeline_uid.pipeline_run_id)
        with pipeline_ops.resume_pipeline(
            m, pipeline, run_id='run0'
        ) as pipeline_state:
          pipeline_state.is_active()
          mock_snapshot.assert_called_once()
          self.assertEqual(
              'run0',  # Should be run0, not run1
              pipeline.runtime_spec.snapshot_settings.base_pipeline_run_strategy.base_run_id,
          )

  def test_revive_pipeline_run(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_id = pipeline.pipeline_info.id
      # Enforce the same run_id
      run_id = pipeline.runtime_spec.pipeline_run_id.field_value.string_value
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['ExampleGen'])

      # Error if attempt to revive the pipeline when there is no previous run.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.revive_pipeline_run(
            m, pipeline_id=pipeline_id, pipeline_run_id=run_id
        )
      self.assertEqual(
          status_lib.Code.NOT_FOUND, exception_context.exception.code
      )

      # Initiate a pipeline start.
      pipeline_state_run1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Error if attempt to revive the pipeline when the run_id is still active.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.revive_pipeline_run(
            m, pipeline_id=pipeline_id, pipeline_run_id=run_id
        )
      self.assertEqual(
          status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
      )

      def _inactivate(pipeline_state):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.CANCELED
            )

      thread = threading.Thread(target=_inactivate, args=(pipeline_state_run1,))
      thread.start()
      # Stop pipeline so we can revive.
      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      )

      with pipeline_state_run1:
        example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
        trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
        with pipeline_state_run1.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run1.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        pipeline_state_run1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.CANCELED
        )
        pipeline_state_run1.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED)
        )
      # Only Trainer is marked to run since ExampleGen succeeded in previous
      # run.
      expected_pipeline = copy.deepcopy(pipeline)
      with pipeline_ops.revive_pipeline_run(
          m, pipeline_id=pipeline_id, pipeline_run_id=run_id
      ) as pipeline_state_run3:
        self.assertEqual(
            pipeline_state_run3.get_node_state(trainer_node_uid).state,
            pstate.NodeState.STARTED,
        )
        self.assertEqual(
            pipeline_state_run3.get_node_state(example_gen_node_uid).state,
            pstate.NodeState.COMPLETE,
        )
        self.assertEqual(expected_pipeline, pipeline_state_run3.pipeline)
        pipeline_state_run3.is_active()

  def test_revive_pipeline_run_with_updated_ir(self):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_id = pipeline.pipeline_info.id
      # Enforce the same run_id
      run_id = pipeline.runtime_spec.pipeline_run_id.field_value.string_value
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'

      # Initiate a pipeline start.
      pipeline_state_run1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _inactivate(pipeline_state):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.CANCELED
            )

      thread = threading.Thread(target=_inactivate, args=(pipeline_state_run1,))
      thread.start()
      # Stop pipeline so we can revive.
      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      )

      with pipeline_state_run1:
        example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
        with pipeline_state_run1.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        pipeline_state_run1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.CANCELED
        )
        pipeline_state_run1.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED)
        )

      pipeline_to_update_to = copy.deepcopy(pipeline)
      pipeline_to_update_to.nodes[
          0
      ].pipeline_node.execution_options.max_execution_retries = 10
      expected_pipeline = copy.deepcopy(pipeline_to_update_to)
      with pipeline_ops.revive_pipeline_run(
          m,
          pipeline_id=pipeline_id,
          pipeline_run_id=run_id,
          pipeline_to_update_with=pipeline_to_update_to,
      ) as pipeline_state_run2:
        self.assertEqual(
            pipeline_state_run2.get_node_state(example_gen_node_uid).state,
            pstate.NodeState.STARTED,
        )
        self.assertEqual(expected_pipeline, pipeline_state_run2.pipeline)
        pipeline_state_run2.is_active()

  def test_revive_pipeline_run_when_concurrent_pipeline_runs_enabled(self):
    with test_utils.concurrent_pipeline_runs_enabled_env():
      with self._mlmd_connection as m:
        pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
        pipeline_id = pipeline.pipeline_info.id
        pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
        node_example_gen = pipeline.nodes.add().pipeline_node
        node_example_gen.node_info.id = 'ExampleGen'
        node_example_gen.downstream_nodes.extend(['Trainer'])
        node_trainer = pipeline.nodes.add().pipeline_node
        node_trainer.node_info.id = 'Trainer'
        node_trainer.upstream_nodes.extend(['ExampleGen'])

        # Initiate a pipeline start.
        pipeline_state_run1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

        with pipeline_state_run1:
          example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
          trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
          with pipeline_state_run1.node_state_update_context(
              example_gen_node_uid
          ) as node_state:
            node_state.update(pstate.NodeState.COMPLETE)
          with pipeline_state_run1.node_state_update_context(
              trainer_node_uid
          ) as node_state:
            node_state.update(pstate.NodeState.FAILED)
          pipeline_state_run1.set_pipeline_execution_state(
              metadata_store_pb2.Execution.CANCELED
          )
          pipeline_state_run1.initiate_stop(
              status_lib.Status(code=status_lib.Code.ABORTED)
          )

        run_id = pipeline.runtime_spec.pipeline_run_id.field_value.string_value

        # Success if pipeline revived with run id.
        self.assertEqual('run0', pipeline_uid.pipeline_run_id)
        with pipeline_ops.revive_pipeline_run(
            m, pipeline_id=pipeline_id, pipeline_run_id=run_id
        ) as pipeline_state_run2:
          pipeline_state_run2.is_active()

  def test_revive_pipeline_run_with_subpipelines(self):
    with self._mlmd_connection as m:
      pipeline = test_sync_pipeline.create_pipeline_with_subpipeline()
      runtime_parameter_utils.substitute_runtime_parameter(
          pipeline,
          {
              constants.PIPELINE_ROOT_PARAMETER_NAME: '/path/to/root',
              constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'run0',
          },
      )
      example_gen = test_utils.get_node(pipeline, 'my_example_gen')
      example_gen_uid = task_lib.NodeUid.from_node(pipeline, example_gen)
      sub_pipeline = test_utils.get_node(pipeline, 'sub-pipeline')
      sub_pipeline_uid = task_lib.NodeUid.from_node(pipeline, sub_pipeline)
      transform = test_utils.get_node(pipeline, 'my_transform')
      transform_uid = task_lib.NodeUid.from_node(pipeline, transform)
      pipeline_state_1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _inactivate(pipeline_state):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.CANCELED
            )

      thread = threading.Thread(target=_inactivate, args=(pipeline_state_1,))
      thread.start()
      # Stop pipeline so we can revive.
      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      )
      # Mark all nodes as STOPPED manually.
      with pipeline_state_1:
        pipeline_state_1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.CANCELED
        )
        with pipeline_state_1.node_state_update_context(
            sub_pipeline_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)
        with pipeline_state_1.node_state_update_context(
            transform_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)

      # Mark example gen as COMPLETE so subpipeline will start.
      with pipeline_state_1:
        with pipeline_state_1.node_state_update_context(
            example_gen_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)

      revived_pipeline_state_1 = pipeline_ops.revive_pipeline_run(
          m,
          pipeline_id=pipeline.pipeline_info.id,
          pipeline_run_id=pipeline.runtime_spec.pipeline_run_id.field_value.string_value,
      )

      with revived_pipeline_state_1:
        node_states_dict = revived_pipeline_state_1.get_node_states_dict()
        self.assertEqual(
            node_states_dict[example_gen_uid].state, pstate.NodeState.COMPLETE
        )
        self.assertEqual(
            node_states_dict[sub_pipeline_uid].state, pstate.NodeState.STARTED
        )
        self.assertEqual(
            node_states_dict[transform_uid].state, pstate.NodeState.STARTED
        )

      # Stop pipeline again.
      thread = threading.Thread(
          target=_inactivate, args=(revived_pipeline_state_1,)
      )
      thread.start()
      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      )

      # Add execution for subpipeline and mark schema_gen as COMPLETE
      sub_pipeline_proto = sub_pipeline.raw_proto()
      subpipeline_state = pipeline_ops.initiate_pipeline_start(
          m, sub_pipeline_proto
      )
      stats_gen = test_utils.get_node(sub_pipeline_proto, 'my_statistics_gen')
      stats_gen_uid = task_lib.NodeUid.from_node(sub_pipeline_proto, stats_gen)
      schema_gen = test_utils.get_node(sub_pipeline_proto, 'my_schema_gen')
      schema_gen_uid = task_lib.NodeUid.from_node(
          sub_pipeline_proto, schema_gen
      )

      with subpipeline_state:
        with subpipeline_state.node_state_update_context(
            stats_gen_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with subpipeline_state.node_state_update_context(
            schema_gen_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)
        subpipeline_execution = subpipeline_state.execution

      # Stop subpipeline.
      thread = threading.Thread(target=_inactivate, args=(subpipeline_state,))
      thread.start()
      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(sub_pipeline_proto)
      )

      # Mark all nodes as STOPPED manually.
      with pipeline_state_1:
        pipeline_state_1.set_pipeline_execution_state(
            metadata_store_pb2.Execution.CANCELED
        )
        with pipeline_state_1.node_state_update_context(
            sub_pipeline_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)
        with pipeline_state_1.node_state_update_context(
            transform_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)

      # Mark the subpipeline execution as CANCELLED
      sub_pipeline_run_id = f'sub-pipeline_run0_{subpipeline_execution.id}'
      with mlmd_state.mlmd_execution_atomic_op(
          m, subpipeline_execution.id
      ) as mlmd_execution:
        mlmd_execution.last_known_state = (
            metadata_store_pb2.Execution.State.CANCELED
        )
        # Update the pipeline run for execution to be appropraite form.
        data_types_utils.set_metadata_value(
            mlmd_execution.custom_properties['pipeline_run_id'],
            sub_pipeline_run_id,
        )
        subpipeline_execution = mlmd_execution
      # Associate subpipeline contexts with
      contexts = context_lib.prepare_contexts(m, sub_pipeline.contexts)
      execution_lib.put_executions(m, [subpipeline_execution], contexts)

      revived_pipeline_state_2 = pipeline_ops.revive_pipeline_run(
          m,
          pipeline_id=pipeline.pipeline_info.id,
          pipeline_run_id=pipeline.runtime_spec.pipeline_run_id.field_value.string_value,
      )

      with revived_pipeline_state_2:
        node_states_dict = revived_pipeline_state_2.get_node_states_dict()
        self.assertEqual(
            node_states_dict[sub_pipeline_uid].state, pstate.NodeState.RUNNING
        )

      with pstate.PipelineState.load(
          m,
          task_lib.PipelineUid.from_pipeline_id_and_run_id(
              sub_pipeline_proto.pipeline_info.id, sub_pipeline_run_id
          ),
      ) as subpipeline_state:
        node_states_dict = subpipeline_state.get_node_states_dict()
        self.assertEqual(
            node_states_dict[stats_gen_uid].state, pstate.NodeState.COMPLETE
        )
        self.assertEqual(
            node_states_dict[schema_gen_uid].state, pstate.NodeState.STARTED
        )

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_invalid_partial_run(
      self, mock_snapshot
  ):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      incorrect_partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['InvalidaNode'],
          to_nodes=['Trainer'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(),
      )
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'specified in from_nodes/to_nodes are not present in the pipeline.',
      ):
        pipeline_ops.initiate_pipeline_start(
            m, pipeline, partial_run_option=incorrect_partial_run_option
        )

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_partial_run(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings
      )
      expected_pipeline.nodes[
          0
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2
      ].pipeline_node.execution_options.run.SetInParent()

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'],
          to_nodes=['Trainer'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(),
      )
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state:
        mock_snapshot.assert_called_once()
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)

  @parameterized.named_parameters(
      dict(
          testcase_name='cache_subpipeline',
          run_subpipeline=False,
      ),
      dict(
          testcase_name='run_subpipeline',
          run_subpipeline=True,
      ),
  )
  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_partial_run_and_subpipeline(
      self, mock_snapshot, run_subpipeline
  ):
    with self._mlmd_connection as m:
      pipeline = test_sync_pipeline.create_pipeline_with_subpipeline()
      runtime_parameter_utils.substitute_runtime_parameter(
          pipeline,
          {
              constants.PIPELINE_ROOT_PARAMETER_NAME: '/my/pipeline/root',
              constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'run-0123',
          },
      )

      expected_pipeline = copy.deepcopy(pipeline)
      example_gen = expected_pipeline.nodes[0].pipeline_node
      subpipeline = expected_pipeline.nodes[1].sub_pipeline
      subpipeline_begin = subpipeline.nodes[0].pipeline_node
      transform = expected_pipeline.nodes[2].pipeline_node
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings
      )

      skip = pipeline_pb2.NodeExecutionOptions.Skip(
          reuse_artifacts_mode=pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      run = pipeline_pb2.NodeExecutionOptions.Run(
          perform_snapshot=True, depends_on_snapshot=True
      )
      example_gen.execution_options.skip.CopyFrom(skip)

      if run_subpipeline:
        subpipeline_begin.execution_options.run.CopyFrom(run)
        transform.execution_options.run.depends_on_snapshot = True
      else:
        subpipeline_begin.execution_options.skip.CopyFrom(skip)
        transform.execution_options.run.CopyFrom(run)

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['sub-pipeline'] if run_subpipeline else ['my_transform'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(),
      )
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state:
        mock_snapshot.assert_called_once()
        self.assertProtoEquals(expected_pipeline, pipeline_state.pipeline)

      if run_subpipeline:
        # If the subpipeline should be run then we should not have pre-loaded a
        # run for it.
        with self.assertRaises(status_lib.StatusNotOkError):
          pstate.PipelineState.load_run(
              m, 'sub-pipeline', 'sub-pipeline_run-0123'
          )
      else:
        # Skipped subpipelines should have a run injected so their nodes are
        # properly marked as cached.
        with pstate.PipelineState.load_run(
            m, 'sub-pipeline', 'sub-pipeline_run-0123'
        ) as subpipeline_state:
          self.assertEqual(
              subpipeline_state.stop_initiated_reason().code, status_lib.Code.OK
          )

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_partial_run_with_previously_failed_nodes(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform', 'Trainer'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['ExampleGen'])

      example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')

      def _stop_pipeline(pipeline_state):
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE
        )
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED)
        )

      # In run0, trainer and transform failed.
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline
      ) as pipeline_state_run0:
        with pipeline_state_run0.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run0.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        with pipeline_state_run0.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        _stop_pipeline(pipeline_state_run0)

      # Partial run based on run0, trainer is skipped and state indicates that
      # it failed previously. Only transform runs and it fails again.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'], to_nodes=['Transform']
      )
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')

      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state_run1:
        self.assertEqual(
            pipeline_state_run1.get_node_state(trainer_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN,
        )
        self.assertEqual(
            pipeline_state_run1.get_node_state(
                trainer_node_uid, pstate._PREVIOUS_NODE_STATES
            ).state,
            pstate.NodeState.FAILED,
        )
        self.assertEqual(
            pipeline_state_run1.get_node_state(example_gen_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN,
        )
        self.assertEqual(
            pipeline_state_run1.get_node_state(
                example_gen_node_uid, pstate._PREVIOUS_NODE_STATES
            ).state,
            pstate.NodeState.COMPLETE,
        )
        self.assertEqual(
            pipeline_state_run1.get_node_state(transform_node_uid).state,
            pstate.NodeState.STARTED,
        )

        with pipeline_state_run1.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)
        _stop_pipeline(pipeline_state_run1)

      # Partial run based on run1, trainer and transform are skipped and
      # correctly indicate they've failed previously.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['ExampleGen'], to_nodes=['ExampleGen']
      )
      pipeline.runtime_spec.pipeline_run_id.field_value.string_value = 'run2'
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state_run2:
        self.assertEqual(
            pipeline_state_run2.get_node_state(trainer_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN,
        )
        self.assertEqual(
            pipeline_state_run2.get_node_state(
                trainer_node_uid, pstate._PREVIOUS_NODE_STATES
            ).state,
            pstate.NodeState.FAILED,
        )
        self.assertEqual(
            pipeline_state_run2.get_node_state(transform_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN,
        )
        self.assertEqual(
            pipeline_state_run2.get_node_state(
                transform_node_uid, pstate._PREVIOUS_NODE_STATES
            ).state,
            pstate.NodeState.FAILED,
        )
        _stop_pipeline(pipeline_state_run2)
      mock_snapshot.assert_called()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_initiate_pipeline_start_with_partial_run_default_to_nodes(
      self, mock_snapshot
  ):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings
      )

      expected_pipeline.nodes[
          0
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2
      ].pipeline_node.execution_options.run.SetInParent()

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'],
          snapshot_settings=partial_run_utils.latest_pipeline_snapshot_settings(),
      )
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state:
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)
        mock_snapshot.assert_called_once()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_partial_run_defaults_to_latest_pipeline_run_strategy(
      self, mock_snapshot
  ):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_transform.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      # partial_run_option without artifact_reuse_strategy should default to
      # latest_pipeline_run_strategy.
      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Transform'], to_nodes=['Trainer']
      )

      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings
      )
      expected_pipeline.nodes[
          0
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.perform_snapshot = True
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2
      ].pipeline_node.execution_options.run.SetInParent()

      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state:
        self.assertEqual(expected_pipeline, pipeline_state.pipeline)
        mock_snapshot.assert_called_once()

  @mock.patch.object(partial_run_utils, 'snapshot')
  def test_partial_run_with_previously_skipped_nodes(self, mock_snapshot):
    with self._mlmd_connection as m:
      pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_example_gen = pipeline.nodes.add().pipeline_node
      node_example_gen.node_info.id = 'ExampleGen'
      node_example_gen.downstream_nodes.extend(['Transform'])
      node_transform = pipeline.nodes.add().pipeline_node
      node_transform.node_info.id = 'Transform'
      node_transform.upstream_nodes.extend(['ExampleGen'])
      node_example_gen.downstream_nodes.extend(['Trainer'])
      node_trainer = pipeline.nodes.add().pipeline_node
      node_trainer.node_info.id = 'Trainer'
      node_trainer.upstream_nodes.extend(['Transform'])

      example_gen_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')

      def _stop_pipeline(pipeline_state):
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE
        )
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.ABORTED)
        )

      with pipeline_ops.initiate_pipeline_start(
          m, pipeline
      ) as pipeline_state_run0:
        with pipeline_state_run0.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        with pipeline_state_run0.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.SKIPPED)
        with pipeline_state_run0.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)
        _stop_pipeline(pipeline_state_run0)

      partial_run_option = pipeline_pb2.PartialRun(
          from_nodes=['Trainer'], to_nodes=['Trainer']
      )
      expected_pipeline = copy.deepcopy(pipeline)
      partial_run_utils.set_latest_pipeline_run_strategy(
          expected_pipeline.runtime_spec.snapshot_settings
      )
      expected_pipeline.nodes[
          0
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
      )
      expected_pipeline.nodes[
          1
      ].pipeline_node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.OPTIONAL
      )
      expected_pipeline.nodes[
          2
      ].pipeline_node.execution_options.run.depends_on_snapshot = True
      expected_pipeline.nodes[
          2
      ].pipeline_node.execution_options.run.perform_snapshot = True
      # Check that SKIPPED node will be marked as OPTIONAL for snapshotting.
      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state_run1:
        self.assertEqual(expected_pipeline, pipeline_state_run1.pipeline)
        self.assertEqual(
            pipeline_state_run1.get_node_state(transform_node_uid).state,
            pstate.NodeState.SKIPPED_PARTIAL_RUN,
        )
        self.assertEqual(
            pipeline_state_run1.get_node_state(
                transform_node_uid, pstate._PREVIOUS_NODE_STATES
            ).state,
            pstate.NodeState.SKIPPED,
        )
        _stop_pipeline(pipeline_state_run1)

      with pipeline_ops.initiate_pipeline_start(
          m, pipeline, partial_run_option=partial_run_option
      ) as pipeline_state_run2:
        self.assertEqual(expected_pipeline, pipeline_state_run2.pipeline)
      mock_snapshot.assert_called()

  def test_update_gets_post_processed(self):
    def _apply_update(pipeline_state):
      # Wait for the pipeline to be in update initiated state.
      while True:
        with pipeline_state:
          if pipeline_state.is_update_initiated():
            break
        time.sleep(0.5)
      # Now apply the update.
      with pipeline_ops._PIPELINE_OPS_LOCK:
        with pipeline_state:
          pipeline_state.apply_pipeline_update()

    with self._mlmd_connection as m:
      with test_utils.prepare_orchestrator_for_pipeline_run_environment():
        pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
        # Initiate a pipeline start.
        pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
        thread = threading.Thread(target=_apply_update, args=(pipeline_state,))
        thread.start()

        updated_pipeline = pipeline_pb2.Pipeline()
        updated_pipeline.CopyFrom(pipeline)
        updated_pipeline.sdk_version = 'some.sdk.version'
        pipeline_ops.update_pipeline(
            m,
            updated_pipeline,
            update_options=pipeline_pb2.UpdateOptions(),
        )

        thread.join()
        # Pipeline gets postprocessed twice, once for start and once for update.
        self.assertEqual(
            pipeline_state.pipeline.sdk_version,
            'postprocessed',
        )

  def test_revive_gets_post_processed(self):
    def _inactivate(pipeline_state):
      time.sleep(2.0)
      with pipeline_ops._PIPELINE_OPS_LOCK:
        with pipeline_state:
          pipeline_state.set_pipeline_execution_state(
              metadata_store_pb2.Execution.CANCELED
          )

    with self._mlmd_connection as m:
      with test_utils.prepare_orchestrator_for_pipeline_run_environment():
        pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
        # Initiate a pipeline start.
        pipeline_state_run1 = pipeline_ops.initiate_pipeline_start(m, pipeline)

        thread = threading.Thread(
            target=_inactivate, args=(pipeline_state_run1,)
        )
        thread.start()
        # Stop pipeline so we can revive.
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid.from_pipeline(pipeline)
        )
        thread.join()
        updated_pipeline = pipeline_pb2.Pipeline()
        updated_pipeline.CopyFrom(pipeline)
        updated_pipeline.sdk_version = 'some.sdk.version'
        pipeline_state = pipeline_ops.revive_pipeline_run(
            m,
            'test_pipeline',
            pipeline_run_id='run0',
            pipeline_to_update_with=updated_pipeline,
        )

        self.assertEqual(
            pipeline_state.pipeline.sdk_version,
            'postprocessed',
        )

  def test_initiate_pipeline_start_gets_post_processed(self):
    with self._mlmd_connection as m:
      with test_utils.prepare_orchestrator_for_pipeline_run_environment():
        pipeline = _test_pipeline('test_pipeline', pipeline_pb2.Pipeline.SYNC)
        pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

        self.assertEqual(
            pipeline_state.pipeline.sdk_version,
            'postprocessed',
        )

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
      ),
  )
  def test_stop_pipeline_non_existent_or_inactive(self, pipeline):
    with self._mlmd_connection as m:
      # Stop pipeline without creating one.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid.from_pipeline(pipeline)
        )
      self.assertEqual(
          status_lib.Code.NOT_FOUND, exception_context.exception.code
      )

      # Stop a non-existent pipeline with ignore_non_existent_or_inactive set
      # should not raise.
      pipeline_ops.stop_pipelines(
          m,
          [task_lib.PipelineUid.from_pipeline(pipeline)],
          ignore_non_existent_or_inactive=True,
      )

      # Initiate pipeline start and mark it completed.
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        pipeline_state.initiate_stop(status_lib.Status(code=status_lib.Code.OK))
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE
        )

      # Try to initiate stop again.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(m, pipeline_uid)
      self.assertEqual(
          status_lib.Code.NOT_FOUND, exception_context.exception.code
      )

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
      ),
  )
  def test_stop_pipeline_wait_for_inactivation(self, pipeline):
    with self._mlmd_connection as m:
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _inactivate(pipeline_state):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.COMPLETE
            )

      thread = threading.Thread(target=_inactivate, args=(pipeline_state,))
      thread.start()

      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline), timeout_secs=20.0
      )

      thread.join()

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
      ),
  )
  def test_stop_pipeline_returns_immediately(self, pipeline):
    with self._mlmd_connection as m:
      mock_wait_for_predicate = self.enter_context(
          mock.patch.object(pipeline_ops, '_wait_for_predicate', autospec=True)
      )
      pipeline_ops.initiate_pipeline_start(m, pipeline)

      pipeline_ops.stop_pipeline(
          m,
          task_lib.PipelineUid.from_pipeline(pipeline),
          timeout_secs=20.0,
          return_immediately=True,
      )
      mock_wait_for_predicate.assert_not_called()

  @parameterized.named_parameters(
      dict(testcase_name='async', pipeline=_test_pipeline('pipeline1')),
      dict(
          testcase_name='sync',
          pipeline=_test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
      ),
  )
  def test_stop_pipeline_wait_for_inactivation_timeout(self, pipeline):
    with self._mlmd_connection as m:
      pipeline_ops.initiate_pipeline_start(m, pipeline)

      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Timed out.*waiting for inactivation of pipelines.',
      ) as exception_context:
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid.from_pipeline(pipeline), timeout_secs=1.0
        )
      self.assertEqual(
          status_lib.Code.DEADLINE_EXCEEDED, exception_context.exception.code
      )

  def test_backfill_node(self):
    pipeline = test_async_pipeline.create_pipeline()

    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    trainer_node_uid = task_lib.NodeUid(
        node_id='my_trainer', pipeline_uid=pipeline_uid
    )

    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)

      # Check - can't backfill a RUNNING node
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        with pipeline_state.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.RUNNING)

      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Can only backfill nodes in a stopped or failed',
      ):
        pipeline_ops.initiate_node_backfill(m, trainer_node_uid)

      # Check - can backfill a STOPPED node
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        with pipeline_state.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STOPPED)
      pipeline_ops.initiate_node_backfill(m, trainer_node_uid)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)
        self.assertNotEqual('', node_state.backfill_token)

  def test_stop_node_wait_for_inactivation(self):
    pipeline = test_async_pipeline.create_pipeline()
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    node_uid = task_lib.NodeUid(node_id='my_trainer', pipeline_uid=pipeline_uid)
    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)

      def _inactivate():
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
            with pipeline_state.node_state_update_context(
                node_uid
            ) as node_state:
              node_state.update(
                  pstate.NodeState.STOPPED,
                  status_lib.Status(code=status_lib.Code.CANCELLED),
              )

      thread = threading.Thread(target=_inactivate, args=())
      thread.start()
      pipeline_ops.stop_node(m, node_uid, timeout_secs=20.0)
      thread.join()

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)

      # Restart node.
      with pipeline_ops.initiate_node_start(m, node_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  def test_stop_node_wait_for_inactivation_timeout(self):
    pipeline = test_async_pipeline.create_pipeline()
    pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
    node_uid = task_lib.NodeUid(node_id='my_trainer', pipeline_uid=pipeline_uid)
    with self._mlmd_connection as m:
      pstate.PipelineState.new(m, pipeline)
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Timed out.*waiting for node inactivation.',
      ) as exception_context:
        pipeline_ops.stop_node(m, node_uid, timeout_secs=1.0)
      self.assertEqual(
          status_lib.Code.DEADLINE_EXCEEDED, exception_context.exception.code
      )

      # Even if `wait_for_inactivation` times out, the node should be in state
      # STOPPING or STOPPED to prevent future triggers.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(node_uid)
        self.assertIn(
            node_state.state,
            (pstate.NodeState.STOPPING, pstate.NodeState.STOPPED),
        )

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_orchestrate_active_pipelines(
      self, mock_async_task_gen, mock_sync_task_gen
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      # Sync and async active pipelines.
      async_pipelines = [
          _test_pipeline('pipeline1'),
          _test_pipeline('pipeline2'),
      ]
      sync_pipelines = [
          _test_pipeline('pipeline3', pipeline_pb2.Pipeline.SYNC),
          _test_pipeline('pipeline4', pipeline_pb2.Pipeline.SYNC),
      ]

      for pipeline in async_pipelines + sync_pipelines:
        pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Active executions for active async pipelines.
      mock_async_task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          async_pipelines[0]
                      ),
                      node_id='Transform',
                  )
              )
          ],
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          async_pipelines[1]
                      ),
                      node_id='Trainer',
                  )
              )
          ],
      ]

      # Active executions for active sync pipelines.
      mock_sync_task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[0]
                      ),
                      node_id='Trainer',
                  )
              )
          ],
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[1]
                      ),
                      node_id='Validator',
                  )
              )
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager,
          task_queue,
          service_jobs.DummyServiceJobManager(),
      )

      self.assertEqual(2, mock_async_task_gen.return_value.generate.call_count)
      self.assertEqual(2, mock_sync_task_gen.return_value.generate.call_count)

      # Verify that tasks are enqueued in the expected order.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(
          test_utils.create_node_uid('pipeline1', 'Transform'), task.node_uid
      )
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(
          test_utils.create_node_uid('pipeline2', 'Trainer'), task.node_uid
      )
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(
          test_utils.create_node_uid('pipeline3', 'Trainer', 'run0'),
          task.node_uid,
      )
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(
          test_utils.create_node_uid('pipeline4', 'Validator', 'run0'),
          task.node_uid,
      )
      self.assertTrue(task_queue.is_empty())

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  @mock.patch.object(
      task_gen_utils, 'generate_cancel_task_from_running_execution'
  )
  def test_orchestrate_stop_initiated_pipelines(
      self,
      pipeline,
      mock_gen_task_from_active,
      mock_async_task_gen,
      mock_sync_task_gen,
  ):
    events = []

    def recorder(event):
      if not isinstance(event, event_observer.PipelineFinished):
        return
      events.append(event)

    with event_observer.init(), self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      event_observer.register_observer(recorder)

      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.CANCELLED)
        )
        pipeline_execution_id = pipeline_state.execution_id

      task_queue = tq.TaskQueue()

      # For the stop-initiated pipeline, "Transform" execution task is in queue,
      # "Trainer" has an active execution in MLMD but no task in queue,
      # "Evaluator" has no active execution.
      task_queue.enqueue(
          test_utils.create_exec_node_task(
              task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Transform',
              )
          )
      )
      transform_task = task_queue.dequeue()  # simulates task being processed
      mock_gen_task_from_active.side_effect = [
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Trainer',
              ),
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
          None,
          None,
          None,
          None,
      ]

      self.assertTrue(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )

      # PipelineFinished event should not trigger since not all the nodes are
      # stopped.
      event_observer.testonly_wait()
      self.assertEqual([], events)

      # There are no active pipelines so these shouldn't be called.
      mock_async_task_gen.assert_not_called()
      mock_sync_task_gen.assert_not_called()

      # stop_node_services should be called for ExampleGen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen'
      )
      self._mock_service_job_manager.reset_mock()

      task_queue.task_done(transform_task)  # Pop out transform task.

      # CancelNodeTask for the "Transform" ExecNodeTask should be next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.CancelNodeTask)
      self.assertEqual('Transform', task.node_uid.node_id)

      # ExecNodeTask (with is_cancelled=True) for "Trainer" is next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual('Trainer', task.node_uid.node_id)
      self.assertEqual(task_lib.NodeCancelType.CANCEL_EXEC, task.cancel_type)

      self.assertTrue(task_queue.is_empty())

      mock_gen_task_from_active.assert_has_calls([
          mock.call(
              m,
              pipeline_state.pipeline,
              node_proto_view.get_view(pipeline.nodes[2].pipeline_node),
              mock.ANY,
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
          mock.call(
              m,
              pipeline_state.pipeline,
              node_proto_view.get_view(pipeline.nodes[3].pipeline_node),
              mock.ANY,
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
      ])
      self.assertEqual(2, mock_gen_task_from_active.call_count)

      # Pipeline execution should continue to be active since active node
      # executions were found in the last call to `orchestrate`.
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertTrue(execution_lib.is_execution_active(execution))

      # Call `orchestrate` again; this time there are no more active node
      # executions so the pipeline should be marked as cancelled.
      self.assertTrue(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )
      self.assertTrue(task_queue.is_empty())
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertEqual(
          metadata_store_pb2.Execution.CANCELED, execution.last_known_state
      )

      # stop_node_services should be called on Transform which is a mixed
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')]
      )

      # Check that all the node states are STOPPED.
      node_states_dict = _get_node_states_dict(execution)
      self.assertLen(node_states_dict, 4)
      self.assertSetEqual(
          set([pstate.NodeState.STOPPED]),
          set(n.state for n in node_states_dict.values()),
      )

      # Check for the PipelineFinished event
      event_observer.testonly_wait()
      self.assertLen(events, 1)
      event = events[0]
      self.assertEqual('pipeline1', event.pipeline_uid.pipeline_id)
      self.assertEqual(
          status_lib.Status(code=status_lib.Code.CANCELLED), event.status
      )

      # Call `orchestrate` again; expecting False as the pipeline is no longer
      # active.
      self.assertFalse(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )

  @mock.patch.object(
      task_gen_utils, 'generate_cancel_task_from_running_execution'
  )
  def test_orchestrate_stop_initiated_pipelines_with_paired_nodes(
      self,
      mock_gen_task_from_active,
  ):
    tmp_dir = self.get_temp_dir()
    pipeline = _test_pipeline(
        pipeline_id='pipeline',
        execution_mode=pipeline_pb2.Pipeline.SYNC,
        pipeline_root=tmp_dir,
    )
    events = []

    def recorder(event):
      if not isinstance(event, event_observer.PipelineFinished):
        return
      events.append(event)

    with event_observer.init(), self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      event_observer.register_observer(recorder)
      paired_start = pipeline.nodes.add().pipeline_node
      paired_start.node_info.id = 'PairedStart'
      doomed_node = pipeline.nodes.add().pipeline_node
      doomed_node.node_info.id = 'DoomedNode'
      paired_end = pipeline.nodes.add().pipeline_node
      paired_end.node_info.id = 'PairedEnd'
      # Add execution type because we didn't compile and need to register the
      # execution.
      paired_end.node_info.type.CopyFrom(
          metadata_store_pb2.ExecutionType(name='PairedEnd')
      )
      paired_end.execution_options.resource_lifetime.lifetime_start = (
          'PairedStart'
      )

      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      paired_start_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='PairedStart'
      )
      doomed_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='DoomedNode'
      )
      paired_end_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='PairedEnd'
      )
      pipeline_ops.initiate_pipeline_start(m, pipeline)

      with pstate.PipelineState.load(
          m,
          pipeline_uid,
      ) as pipeline_state:
        pipeline_state.initiate_stop(
            status_lib.Status(code=status_lib.Code.CANCELLED)
        )
        pipeline_execution_id = pipeline_state.execution_id
        # PairedStart is COMPLETE
        with pipeline_state.node_state_update_context(
            paired_start_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)
        # DoomedNode is RUNNING
        with pipeline_state.node_state_update_context(
            doomed_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.FAILED)

      task_queue = tq.TaskQueue()
      # For the stop initiated pipeline, PairedStart is complete, DoomedNode is
      # enqueued and wil be canceled, and PairedEnd has no executions.
      task_queue.enqueue(
          test_utils.create_exec_node_task(node_uid=doomed_node_uid)
      )
      doomed_task = task_queue.dequeue()  # simulates task being processed
      self.assertIsInstance(doomed_task, task_lib.ExecNodeTask)
      self.assertEqual(doomed_task.node_uid, doomed_node_uid)
      mock_gen_task_from_active.side_effect = [
          test_utils.create_exec_node_task(
              node_uid=doomed_node_uid,
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
      ]

      self.assertTrue(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )

      # PipelineFinished event should not trigger since not all the nodes are
      # stopped.
      event_observer.testonly_wait()
      self.assertEqual([], events)

      task_queue.task_done(doomed_task)  # Pop out transform task.

      self.assertTrue(task_queue.is_empty())

      # Pipeline execution should continue to be active since PairedEnd is still
      # "active" and so the check for all nodes being stopped is not true.
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertTrue(execution_lib.is_execution_active(execution))

      # Mark PairedEnd as inative to finalize pipeline cleanup.
      with pstate.PipelineState.load(
          m,
          pipeline_uid,
      ) as pipeline_state:
        with pipeline_state.node_state_update_context(
            paired_end_uid
        ) as node_state:
          node_state.update(pstate.NodeState.COMPLETE)

      # Call `orchestrate` again; this time there are no more active node
      # executions so the pipeline should be marked as cancelled.
      self.assertTrue(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )
      self.assertTrue(task_queue.is_empty())
      [execution] = m.store.get_executions_by_id([pipeline_execution_id])
      self.assertEqual(
          metadata_store_pb2.Execution.CANCELED, execution.last_known_state
      )

      # Check that all the node states are STOPPED.
      node_states_dict = _get_node_states_dict(execution)
      self.assertLen(node_states_dict, 3)
      self.assertEqual(
          node_states_dict['PairedStart'].state, pstate.NodeState.COMPLETE
      )
      self.assertEqual(
          node_states_dict['DoomedNode'].state, pstate.NodeState.FAILED
      )
      self.assertEqual(
          node_states_dict['PairedEnd'].state, pstate.NodeState.COMPLETE
      )

      # Check for the PipelineFinished event
      event_observer.testonly_wait()
      self.assertLen(events, 1)
      event = events[0]
      self.assertEqual('pipeline', event.pipeline_uid.pipeline_id)
      self.assertEqual(
          status_lib.Status(code=status_lib.Code.CANCELLED), event.status
      )

      # Call `orchestrate` again; expecting False as the pipeline is no longer
      # active.
      self.assertFalse(
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              task_queue,
              self._mock_service_job_manager,
          )
      )

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  def test_orchestrate_update_initiated_pipelines(self, pipeline):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      task_queue = tq.TaskQueue()

      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task_queue.enqueue(
            test_utils.create_exec_node_task(
                task_lib.NodeUid(
                    pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                    node_id=node_id,
                )
            )
        )
      pipeline_state = pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL
          ),
      )
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      # stop_node_services should be called for ExampleGen.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'ExampleGen')]
      )
      self._mock_service_job_manager.reset_mock()

      # Simulate completion of all the exec node tasks.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.ExecNodeTask)
        self.assertEqual(node_id, task.node_uid.node_id)

      # Verify that cancellation tasks were enqueued in the last `orchestrate`
      # call, and dequeue them.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.CancelNodeTask)
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertEqual(task.cancel_type, task_lib.NodeCancelType.CANCEL_EXEC)
      self.assertTrue(task_queue.is_empty())

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      # stop_node_services should be called for Transform.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')]
      )

      # Check that the node states are STARTING.
      [execution] = m.store.get_executions_by_id([pipeline_state.execution_id])
      node_states_dict = _get_node_states_dict(execution)
      self.assertLen(node_states_dict, 4)
      self.assertSetEqual(
          set([pstate.NodeState.STARTED]),
          set(n.state for n in node_states_dict.values()),
      )

      # Pipeline should no longer be in update-initiated state but be active.
      with pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())

  def test_orchestrate_update_initiated_pipelines_options(self):
    pipeline = _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      task_queue = tq.TaskQueue()

      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task_queue.enqueue(
            test_utils.create_exec_node_task(
                task_lib.NodeUid(
                    pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                    node_id=node_id,
                )
            )
        )
      pipeline_state = pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.PARTIAL,
              reload_nodes=['Transform', 'Trainer'],
          ),
      )
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      # stop_node_services should not be called for ExampleGen since it is not
      # reloaded according to the options.
      self._mock_service_job_manager.stop_node_services.assert_not_called()

      # Simulate completion of all the exec node tasks except evaluator.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.ExecNodeTask)
        self.assertEqual(node_id, task.node_uid.node_id)

      # Verify that cancellation tasks were enqueued in the last `orchestrate`
      # call, and dequeue them.
      for node_id in ('Transform', 'Trainer'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.CancelNodeTask)
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertEqual(task.cancel_type, task_lib.NodeCancelType.CANCEL_EXEC)

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')]
      )

      # Pipeline should no longer be in update-initiated state but be active.
      with pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())

      self.assertTrue(task_queue.is_empty())

  def test_update_pipeline_waits_for_update_application(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline1')
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

      def _apply_update(pipeline_state):
        # Wait for the pipeline to be in update initiated state.
        while True:
          with pipeline_state:
            if pipeline_state.is_update_initiated():
              break
          time.sleep(0.5)
        # Now apply the update.
        with pipeline_ops._PIPELINE_OPS_LOCK:
          with pipeline_state:
            pipeline_state.apply_pipeline_update()

      thread = threading.Thread(target=_apply_update, args=(pipeline_state,))
      thread.start()
      pipeline_ops.update_pipeline(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL
          ),
          timeout_secs=10.0,
      )
      thread.join()

  def test_update_pipeline_wait_for_update_timeout(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError, 'Timed out.*waiting for pipeline update'
      ):
        pipeline_ops.update_pipeline(
            m,
            pipeline,
            update_options=pipeline_pb2.UpdateOptions(
                reload_policy=pipeline_pb2.UpdateOptions.ALL
            ),
            timeout_secs=3.0,
        )

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(
      task_gen_utils, 'generate_cancel_task_from_running_execution'
  )
  def test_orchestrate_update_initiated_pipelines_preempted(
      self,
      pipeline,
      mock_gen_task_from_active,
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      task_queue = tq.TaskQueue()

      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task_queue.enqueue(
            test_utils.create_exec_node_task(
                task_lib.NodeUid(
                    pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                    node_id=node_id,
                )
            )
        )
      pipeline_state = pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL
          ),
      )
      with pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())

      # Assume orchestator is preemplted at this point.
      # task_queue is empty after the orchestator is restarted.
      task_queue = tq.TaskQueue()
      self.assertTrue(task_queue.is_empty())

      mock_gen_task_from_active.side_effect = [
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Transform',
              ),
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Trainer',
              ),
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid.from_pipeline(pipeline),
                  node_id='Evaluator',
              ),
              cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
          ),
          None,
          None,
          None,
      ]

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      # stop_node_services should be called for ExampleGen.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'ExampleGen')]
      )
      self._mock_service_job_manager.reset_mock()

      # Verify that cancellation tasks were enqueued in the last `orchestrate`
      # call, and dequeue them.
      for node_id in ('Transform', 'Trainer', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.ExecNodeTask)
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertEqual(task.cancel_type, task_lib.NodeCancelType.CANCEL_EXEC)
      self.assertTrue(task_queue.is_empty())

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      # stop_node_services should be called for Transform.
      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'Transform')]
      )

      # Check that the node states are STARTING.
      [execution] = m.store.get_executions_by_id([pipeline_state.execution_id])
      node_states_dict = _get_node_states_dict(execution)
      self.assertLen(node_states_dict, 4)
      self.assertSetEqual(
          set([pstate.NodeState.STARTED]),
          set(n.state for n in node_states_dict.values()),
      )

      # Pipeline should no longer be in update-initiated state but be active.
      with pipeline_state:
        self.assertFalse(pipeline_state.is_update_initiated())
        self.assertTrue(pipeline_state.is_active())

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  @mock.patch.object(
      task_gen_utils, 'generate_cancel_task_from_running_execution'
  )
  def test_active_pipelines_with_stopped_nodes(
      self,
      pipeline,
      mock_gen_task_from_active,
      mock_async_task_gen,
      mock_sync_task_gen,
  ):
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      mock_task_gen = mock_sync_task_gen
    else:
      mock_task_gen = mock_async_task_gen

    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'

      example_gen_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[0].pipeline_node
      )

      transform_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[1].pipeline_node
      )
      transform_task = test_utils.create_exec_node_task(
          node_uid=transform_node_uid
      )

      trainer_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[2].pipeline_node
      )
      trainer_task = test_utils.create_exec_node_task(node_uid=trainer_node_uid)

      evaluator_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[3].pipeline_node
      )
      evaluator_task = test_utils.create_exec_node_task(
          node_uid=evaluator_node_uid
      )
      cancelled_evaluator_task = test_utils.create_exec_node_task(
          node_uid=evaluator_node_uid,
          cancel_type=task_lib.NodeCancelType.CANCEL_EXEC,
      )

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        # Stop example-gen, trainer and evaluator.
        with pipeline_state.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )
        with pipeline_state.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )
        with pipeline_state.node_state_update_context(
            evaluator_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.ABORTED),
          )

      task_queue = tq.TaskQueue()

      # Simulate a new transform execution being triggered.
      mock_task_gen.return_value.generate.return_value = [transform_task]
      # Simulate ExecNodeTask for trainer already present in the task queue.
      task_queue.enqueue(trainer_task)
      # Simulate Evaluator having an active execution in MLMD.
      mock_gen_task_from_active.side_effect = [evaluator_task]

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      self.assertEqual(1, mock_task_gen.return_value.generate.call_count)

      # stop_node_services should be called on example-gen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'ExampleGen'
      )

      # Verify that tasks are enqueued in the expected order:

      # Pre-existing trainer task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(trainer_task, task)

      # CancelNodeTask for trainer.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.CancelNodeTask)
      self.assertEqual(trainer_node_uid, task.node_uid)

      # ExecNodeTask with is_cancelled=True for evaluator.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(cancelled_evaluator_task, task)

      # ExecNodeTask for newly triggered transform node.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(transform_task, task)

      # No more tasks.
      self.assertTrue(task_queue.is_empty())

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_handling_finalize_pipeline_task(self, task_gen):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC)
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      finalize_reason = status_lib.Status(
          code=status_lib.Code.ABORTED, message='foo bar'
      )
      task_gen.return_value.generate.side_effect = [
          [
              task_lib.FinalizePipelineTask(
                  pipeline_uid=pipeline_uid, status=finalize_reason
              )
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager,
          task_queue,
          service_jobs.DummyServiceJobManager(),
      )
      task_gen.return_value.generate.assert_called_once()
      self.assertTrue(task_queue.is_empty())

      # Load pipeline state and verify stop initiation.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        self.assertEqual(
            finalize_reason, pipeline_state.stop_initiated_reason()
        )

  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_handling_finalize_node_task(self, task_gen):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline1')
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      transform_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='Transform'
      )
      trainer_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='Trainer'
      )
      task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(transform_node_uid),
              task_lib.UpdateNodeStateTask(
                  node_uid=trainer_node_uid, state=pstate.NodeState.FAILED
              ),
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager,
          task_queue,
          service_jobs.DummyServiceJobManager(),
      )
      task_gen.return_value.generate.assert_called_once()
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(transform_node_uid, task.node_uid)

      # Load pipeline state and verify trainer node state.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.FAILED, node_state.state)

  def test_error_translated_to_StatusNotOkError(self):
    @pipeline_ops._pipeline_op(lock=False)
    def fn1():
      raise RuntimeError('test error 1')

    @pipeline_ops._pipeline_op(lock=False)
    def fn2():
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS, message='test error 2'
      )

    with self.assertRaisesRegex(
        status_lib.StatusNotOkError, 'test error 1'
    ) as ctxt:
      fn1()
    self.assertEqual(status_lib.Code.UNKNOWN, ctxt.exception.code)

    with self.assertRaisesRegex(
        status_lib.StatusNotOkError, 'test error 2'
    ) as ctxt:
      fn2()
    self.assertEqual(status_lib.Code.ALREADY_EXISTS, ctxt.exception.code)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_executor_node_stop_then_start_flow(
      self, pipeline, mock_async_task_gen, mock_sync_task_gen
  ):
    service_job_manager = service_jobs.DummyServiceJobManager()
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      trainer_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[0].pipeline_node
      )

      # Start pipeline and stop trainer.
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        with pipeline_state.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )

      task_queue = tq.TaskQueue()

      # Simulate ExecNodeTask for trainer already present in the task queue.
      trainer_task = test_utils.create_exec_node_task(node_uid=trainer_node_uid)
      task_queue.enqueue(trainer_task)

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, service_job_manager
      )

      # Dequeue pre-existing trainer task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(trainer_task, task)

      # Dequeue CancelNodeTask for trainer.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.CancelNodeTask)
      self.assertEqual(trainer_node_uid, task.node_uid)

      self.assertTrue(task_queue.is_empty())

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, service_job_manager
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, trainer_node_uid)
      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, service_job_manager
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(trainer_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @parameterized.named_parameters(
      dict(
          testcase_name='async', pipeline=test_async_pipeline.create_pipeline()
      ),
      dict(
          testcase_name='sync',
          pipeline=test_sync_pipeline.create_pipeline(),
      ),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_pure_service_node_stop_then_start_flow(
      self,
      mock_async_task_gen,
      mock_sync_task_gen,
      pipeline,
  ):
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'test-pipeline-run',
        },
    )
    self._mock_service_job_manager.is_pure_service_node.side_effect = (
        lambda _, node_id: node_id == 'my_example_gen'
    )
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      example_gen = pipeline.nodes[0].pipeline_node
      example_gen_node_uid = task_lib.NodeUid.from_node(pipeline, example_gen)

      pipeline_ops.initiate_pipeline_start(m, pipeline)

      test_utils.fake_example_gen_execution_with_state(
          m,
          example_gen,
          metadata_store_pb2.Execution.State.RUNNING,
      )

      eg_execs = m.store.get_executions_by_type(example_gen.node_info.type.name)
      self.assertLen(eg_execs, 1)
      self.assertEqual(
          metadata_store_pb2.Execution.State.RUNNING,
          eg_execs[0].last_known_state,
      )
      execution_lib.register_output_artifacts(
          m, eg_execs[0].id, {'Examples': [standard_artifacts.Examples()]}
      )
      eg_artifact = execution_lib.get_pending_output_artifacts(
          m, eg_execs[0].id
      )
      self.assertEqual(
          types.artifact.ArtifactState.PENDING, eg_artifact['Examples'][0].state
      )

      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        with pipeline_state.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )

      task_queue = tq.TaskQueue()

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      # stop_node_services should be called for ExampleGen which is a pure
      # service node.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'my_example_gen'
      )
      eg_execs = m.store.get_executions_by_type(example_gen.node_info.type.name)
      self.assertLen(eg_execs, 1)
      self.assertEqual(
          metadata_store_pb2.Execution.State.CANCELED,
          eg_execs[0].last_known_state,
      )
      eg_artifact = execution_lib.get_pending_output_artifacts(
          m, eg_execs[0].id
      )
      self.assertEqual(
          types.artifact.ArtifactState.ABANDONED,
          eg_artifact['Examples'][0].state,
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, example_gen_node_uid)
      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_mixed_service_node_stop_then_start_flow(
      self, pipeline, mock_async_task_gen, mock_sync_task_gen
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'

      transform_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[0].pipeline_node
      )

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        # Stop Transform.
        with pipeline_state.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )

      task_queue = tq.TaskQueue()

      # Simulate ExecNodeTask for Transform already present in the task queue.
      transform_task = test_utils.create_exec_node_task(
          node_uid=transform_node_uid
      )
      task_queue.enqueue(transform_task)

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      # stop_node_services should not be called as there was an active
      # ExecNodeTask for Transform which is a mixed service node.
      self._mock_service_job_manager.stop_node_services.assert_not_called()

      # Dequeue pre-existing transform task.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertEqual(transform_task, task)

      # Dequeue CancelNodeTask for transform.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.CancelNodeTask)
      self.assertEqual(transform_node_uid, task.node_uid)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      # stop_node_services should be called for Transform which is a mixed
      # service node and corresponding ExecNodeTask has been dequeued.
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'Transform'
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPED, node_state.state)
        self.assertEqual(status_lib.Code.CANCELLED, node_state.status.code)

      pipeline_ops.initiate_node_start(m, transform_node_uid)
      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STARTED, node_state.state)

  @mock.patch.object(time, 'sleep')
  def test_wait_for_predicate_timeout_secs_None(self, mock_sleep):
    predicate_fn = mock.Mock()
    predicate_fn.side_effect = [False, False, False, True]
    pipeline_ops._wait_for_predicate(predicate_fn, 'testing', 1.0, None)
    self.assertEqual(predicate_fn.call_count, 4)
    self.assertEqual(mock_sleep.call_count, 3)
    predicate_fn.reset_mock()
    mock_sleep.reset_mock()

    predicate_fn.side_effect = [False, False, ValueError('test error')]
    with self.assertRaisesRegex(ValueError, 'test error'):
      pipeline_ops._wait_for_predicate(predicate_fn, 'testing', 1.0, None)
    self.assertEqual(predicate_fn.call_count, 3)
    self.assertEqual(mock_sleep.call_count, 2)

  def test_resume_manual_node(self):
    pipeline = test_manual_node.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'test-pipeline-run',
        },
    )
    manual_node = pipeline.nodes[0].pipeline_node
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pstate.PipelineState.new(m, pipeline)
      contexts = context_lib.prepare_contexts(m, manual_node.contexts)
      execution = execution_publish_utils.register_execution(
          m, manual_node.node_info.type, contexts
      )

      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=m, execution_id=execution.id
      ) as execution:
        node_state_mlmd_value = execution.custom_properties.get(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY
        )
        node_state = manual_task_scheduler.ManualNodeState.from_mlmd_value(
            node_state_mlmd_value
        )
      self.assertEqual(
          node_state.state, manual_task_scheduler.ManualNodeState.WAITING
      )

      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      node_uid = task_lib.NodeUid(
          node_id=manual_node.node_info.id, pipeline_uid=pipeline_uid
      )

      pipeline_ops.resume_manual_node(m, node_uid)

      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=m, execution_id=execution.id
      ) as execution:
        node_state_mlmd_value = execution.custom_properties.get(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY
        )
        node_state = manual_task_scheduler.ManualNodeState.from_mlmd_value(
            node_state_mlmd_value
        )
      self.assertEqual(
          node_state.state, manual_task_scheduler.ManualNodeState.COMPLETED
      )

  @mock.patch.object(pipeline_ops, '_cancel_executions')
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_update_node_state_tasks_handling(
      self, mock_sync_task_gen, mock_cancel_executions
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC
      )
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      eg_node_uid = task_lib.NodeUid(pipeline_uid, 'ExampleGen')
      transform_node_uid = task_lib.NodeUid(pipeline_uid, 'Transform')
      trainer_node_uid = task_lib.NodeUid(pipeline_uid, 'Trainer')
      evaluator_node_uid = task_lib.NodeUid(pipeline_uid, 'Evaluator')

      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state:
        # Set initial states for the nodes.
        with pipeline_state.node_state_update_context(
            eg_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.RUNNING)
        with pipeline_state.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STARTED)
        with pipeline_state.node_state_update_context(
            trainer_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STARTED)
        with pipeline_state.node_state_update_context(
            evaluator_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.RUNNING)

      mock_sync_task_gen.return_value.generate.side_effect = [
          [
              task_lib.UpdateNodeStateTask(
                  node_uid=eg_node_uid, state=pstate.NodeState.COMPLETE
              ),
              task_lib.UpdateNodeStateTask(
                  node_uid=trainer_node_uid, state=pstate.NodeState.RUNNING
              ),
              task_lib.UpdateNodeStateTask(
                  node_uid=evaluator_node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.ABORTED, message='foobar error'
                  ),
              ),
          ],
      ]
      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager,
          task_queue,
          service_jobs.DummyServiceJobManager(),
      )
      self.assertEqual(1, mock_sync_task_gen.return_value.generate.call_count)
      self.assertEqual(1, mock_cancel_executions.call_count)

      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        self.assertEqual(
            pstate.NodeState.COMPLETE,
            pipeline_state.get_node_state(eg_node_uid).state,
        )
        self.assertEqual(
            pstate.NodeState.STARTED,
            pipeline_state.get_node_state(transform_node_uid).state,
        )
        self.assertEqual(
            pstate.NodeState.RUNNING,
            pipeline_state.get_node_state(trainer_node_uid).state,
        )
        self.assertEqual(
            pstate.NodeState.FAILED,
            pipeline_state.get_node_state(evaluator_node_uid).state,
        )
        self.assertEqual(
            status_lib.Status(
                code=status_lib.Code.ABORTED, message='foobar error'
            ),
            pipeline_state.get_node_state(evaluator_node_uid).status,
        )

  @parameterized.parameters(
      _test_pipeline('pipeline1'),
      _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
  )
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_stop_node_services_failure(
      self, pipeline, mock_async_task_gen, mock_sync_task_gen
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'

      example_gen_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[0].pipeline_node
      )
      transform_node_uid = task_lib.NodeUid.from_node(
          pipeline, pipeline.nodes[1].pipeline_node
      )

      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        with pipeline_state.node_state_update_context(
            example_gen_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )
        with pipeline_state.node_state_update_context(
            transform_node_uid
        ) as node_state:
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )

      task_queue = tq.TaskQueue()

      # Simulate failure of stop_node_services.
      self._mock_service_job_manager.stop_node_services.return_value = False

      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )

      self._mock_service_job_manager.stop_node_services.assert_has_calls(
          [mock.call(mock.ANY, 'ExampleGen'), mock.call(mock.ANY, 'Transform')],
          any_order=True,
      )

      # Node state should be STOPPING, not STOPPED since stop_node_services
      # failed.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(example_gen_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.STOPPING, node_state.state)

  @mock.patch.object(pipeline_ops, '_cancel_executions')
  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_stop_node_services_called_for_mixed_service_node_in_terminal_state(
      self, task_gen, mock_cancel_executions
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC
      )
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      transform_node_uid = task_lib.NodeUid(
          pipeline_uid=pipeline_uid, node_id='Transform'
      )
      task_gen.return_value.generate.side_effect = [
          [
              task_lib.UpdateNodeStateTask(
                  node_uid=transform_node_uid, state=pstate.NodeState.FAILED
              ),
          ],
      ]
      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager, task_queue, self._mock_service_job_manager
      )
      task_gen.return_value.generate.assert_called_once()
      self._mock_service_job_manager.stop_node_services.assert_called_once_with(
          mock.ANY, 'Transform'
      )
      self.assertEqual(1, mock_cancel_executions.call_count)

      # Load pipeline state and verify Transform node state.
      with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
        node_state = pipeline_state.get_node_state(transform_node_uid)
        self.assertEqual(pstate.NodeState.FAILED, node_state.state)

  def test_pipeline_run_deadline_exceeded(self):
    class _TestEnv(env._DefaultEnv):
      """TestEnv returns orchestration_options with 1 sec deadline."""

      def get_orchestration_options(self, pipeline):
        return orchestration_options.OrchestrationOptions(deadline_secs=1)

    with _TestEnv():
      with self._mlmd_cm as mlmd_connection_manager:
        m = mlmd_connection_manager.primary_mlmd_handle
        pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
        pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
        pipeline_ops.initiate_pipeline_start(m, pipeline)
        time.sleep(3)  # To trigger the deadline.
        pipeline_ops.orchestrate(
            mlmd_connection_manager,
            tq.TaskQueue(),
            self._mock_service_job_manager,
        )
        with pstate.PipelineState.load(m, pipeline_uid) as pipeline_state:
          self.assertTrue(pipeline_state.is_stop_initiated())
          status = pipeline_state.stop_initiated_reason()
          self.assertEqual(status_lib.Code.DEADLINE_EXCEEDED, status.code)
          self.assertEqual(
              'Pipeline aborted due to exceeding deadline (1 secs)',
              status.message,
          )

  def test_skip_nodes(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline(
          'pipeline1', execution_mode=pipeline_pb2.Pipeline.SYNC
      )
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
      pipeline.nodes.add().pipeline_node.node_info.id = 'ExampleGen'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      pipeline.nodes.add().pipeline_node.node_info.id = 'ModelValidator'
      pipeline.nodes.add().pipeline_node.node_info.id = 'Pusher'
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      pipeline_ops.skip_nodes(
          m,
          [
              task_lib.NodeUid(pipeline_uid, 'Transform'),
              task_lib.NodeUid(pipeline_uid, 'Evaluator'),
          ],
      )
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        states_dict = pipeline_state.get_node_states_dict()
        for node_id in ('ExampleGen', 'Trainer', 'ModelValidator', 'Pusher'):
          self.assertEqual(
              pstate.NodeState.STARTED,
              states_dict[task_lib.NodeUid(pipeline_uid, node_id)].state,
          )
        for node_id in ('Transform', 'Evaluator'):
          self.assertEqual(
              pstate.NodeState.SKIPPED,
              states_dict[task_lib.NodeUid(pipeline_uid, node_id)].state,
          )

        # Change state of Trainer node to RUNNING.
        with pipeline_state.node_state_update_context(
            task_lib.NodeUid(pipeline_uid, 'Trainer')
        ) as node_state:
          node_state.state = pstate.NodeState.RUNNING

      # Calling skip_nodes for Trainer should raise an error as the node is in
      # state RUNNING.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.skip_nodes(
            m,
            [
                task_lib.NodeUid(pipeline_uid, 'Trainer'),
                task_lib.NodeUid(pipeline_uid, 'Pusher'),
            ],
        )
      self.assertEqual(
          status_lib.Code.FAILED_PRECONDITION, exception_context.exception.code
      )
      with pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(pipeline)
      ) as pipeline_state:
        states_dict = pipeline_state.get_node_states_dict()
        self.assertEqual(
            pstate.NodeState.RUNNING,
            states_dict[task_lib.NodeUid(pipeline_uid, 'Trainer')].state,
        )
        self.assertEqual(
            pstate.NodeState.STARTED,
            states_dict[task_lib.NodeUid(pipeline_uid, 'Pusher')].state,
        )

  def test_exception_while_orchestrating_active_pipeline(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      with mock.patch.object(
          pipeline_ops, '_orchestrate_active_pipeline'
      ) as mock_orchestrate_active_pipeline:
        mock_orchestrate_active_pipeline.side_effect = Exception('test error')
        pipeline_ops.orchestrate(
            mlmd_connection_manager,
            tq.TaskQueue(),
            self._mock_service_job_manager,
        )
        mock_orchestrate_active_pipeline.assert_called_once()
        # Verify that the active pipeline is stop-initiated.
        with pipeline_state:
          self.assertTrue(pipeline_state.is_stop_initiated())

  def test_exception_while_orchestrating_stop_initiated_pipeline(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
      with pipeline_ops.initiate_pipeline_start(m, pipeline) as pipeline_state:
        pipeline_state.initiate_stop(
            status_lib.Status(
                code=status_lib.Code.CANCELLED, message='test cancellation'
            )
        )
        self.assertTrue(pipeline_state.is_stop_initiated())
      with mock.patch.object(
          pipeline_ops, '_orchestrate_stop_initiated_pipeline'
      ) as mock_orchestrate_stop_initiated_pipeline:
        mock_orchestrate_stop_initiated_pipeline.side_effect = Exception(
            'test error'
        )
        pipeline_ops.orchestrate(
            mlmd_connection_manager,
            tq.TaskQueue(),
            self._mock_service_job_manager,
        )
        # No exception should be raised.
        mock_orchestrate_stop_initiated_pipeline.assert_called_once()

  def test_exception_while_orchestrating_update_initiated_pipeline(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_ops.initiate_pipeline_start(m, pipeline)
      with pipeline_ops._initiate_pipeline_update(
          m,
          pipeline,
          update_options=pipeline_pb2.UpdateOptions(
              reload_policy=pipeline_pb2.UpdateOptions.ALL
          ),
      ) as pipeline_state:
        self.assertTrue(pipeline_state.is_update_initiated())
      with mock.patch.object(
          pipeline_ops, '_orchestrate_update_initiated_pipeline'
      ) as mock_orchestrate_update_initiated_pipeline:
        mock_orchestrate_update_initiated_pipeline.side_effect = Exception(
            'test error'
        )
        pipeline_ops.orchestrate(
            mlmd_connection_manager,
            tq.TaskQueue(),
            self._mock_service_job_manager,
        )
        mock_orchestrate_update_initiated_pipeline.assert_called_once()
        # Verify that the update-initiated pipeline is stop-initiated.
        with pipeline_state:
          self.assertTrue(pipeline_state.is_stop_initiated())

  def test_exception_while_stop_initiating_on_internal_error(self):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC)
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
      with mock.patch.object(
          pipeline_ops, '_orchestrate_active_pipeline'
      ) as mock_orchestrate_active_pipeline:
        with mock.patch.object(
            pstate.PipelineState, 'initiate_stop'
        ) as mock_initiate_stop:
          mock_orchestrate_active_pipeline.side_effect = Exception('test error')
          mock_initiate_stop.side_effect = Exception('test error 2')
          pipeline_ops.orchestrate(
              mlmd_connection_manager,
              tq.TaskQueue(),
              self._mock_service_job_manager,
          )
          mock_orchestrate_active_pipeline.assert_called_once()
          mock_initiate_stop.assert_called_once()
          # Verify that the active pipeline is not stop-initiated but no
          # exception should be raised.
          with pipeline_state:
            self.assertFalse(pipeline_state.is_stop_initiated())

  def test_start_concurrent_pipeline_runs(self):
    with test_utils.concurrent_pipeline_runs_enabled_env():
      with self._mlmd_cm as mlmd_connection_manager:
        m = mlmd_connection_manager.primary_mlmd_handle
        pipeline1 = _test_pipeline(
            'pipeline', pipeline_pb2.Pipeline.SYNC, 'run0'
        )
        pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline1)
        self.assertEqual(
            pipeline_state.pipeline_uid,
            task_lib.PipelineUid('pipeline', 'run0'),
        )

        # Should be possible to start a new run with a different run id.
        pipeline2 = copy.deepcopy(pipeline1)
        pipeline2.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
        pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline2)
        self.assertEqual(
            pipeline_state.pipeline_uid,
            task_lib.PipelineUid('pipeline', 'run1'),
        )

        # Starting a concurrent run with a duplicate id is prohibited.
        pipeline3 = copy.deepcopy(pipeline2)
        with self.assertRaises(
            status_lib.StatusNotOkError
        ) as exception_context:
          pipeline_ops.initiate_pipeline_start(m, pipeline3)
        self.assertEqual(
            status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
        )

  def test_start_concurrent_pipeline_runs_when_disabled(self) -> bool:
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      pipeline1 = _test_pipeline('pipeline', pipeline_pb2.Pipeline.SYNC, 'run0')
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertEqual(
          pipeline_state.pipeline_uid, task_lib.PipelineUid('pipeline', 'run0')
      )

      # Starting a concurrent run with a different run id is prohibited.
      pipeline2 = copy.deepcopy(pipeline1)
      pipeline2.runtime_spec.pipeline_run_id.field_value.string_value = 'run1'
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.initiate_pipeline_start(m, pipeline2)
      self.assertEqual(
          status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
      )

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_orchestrate_concurrent_pipeline_runs(self, mock_sync_task_gen):
    with test_utils.concurrent_pipeline_runs_enabled_env():
      with self._mlmd_cm as mlmd_connection_manager:
        m = mlmd_connection_manager.primary_mlmd_handle
        # Sync pipelines with same pipeline_id but different run ids.
        sync_pipelines = [
            _test_pipeline(
                'pipeline1', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run0'
            ),
            _test_pipeline(
                'pipeline1', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run1'
            ),
        ]

        for pipeline in sync_pipelines:
          pipeline_ops.initiate_pipeline_start(m, pipeline)

        # Active executions for active sync pipelines.
        mock_sync_task_gen.return_value.generate.side_effect = [
            [
                test_utils.create_exec_node_task(
                    task_lib.NodeUid(
                        pipeline_uid=task_lib.PipelineUid.from_pipeline(
                            sync_pipelines[0]
                        ),
                        node_id='Trainer',
                    )
                )
            ],
            [
                test_utils.create_exec_node_task(
                    task_lib.NodeUid(
                        pipeline_uid=task_lib.PipelineUid.from_pipeline(
                            sync_pipelines[1]
                        ),
                        node_id='Validator',
                    )
                )
            ],
        ]

        task_queue = tq.TaskQueue()
        pipeline_ops.orchestrate(
            mlmd_connection_manager,
            task_queue,
            service_jobs.DummyServiceJobManager(),
        )

        self.assertEqual(2, mock_sync_task_gen.return_value.generate.call_count)

        # Verify that tasks are enqueued in the expected order.
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.ExecNodeTask)
        self.assertEqual(
            test_utils.create_node_uid(
                'pipeline1', 'Trainer', pipeline_run_id='run0'
            ),
            task.node_uid,
        )
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertIsInstance(task, task_lib.ExecNodeTask)
        self.assertEqual(
            test_utils.create_node_uid(
                'pipeline1', 'Validator', pipeline_run_id='run1'
            ),
            task.node_uid,
        )
        self.assertTrue(task_queue.is_empty())

  def test_mixing_concurrent_runs_and_async_pipeline(self):
    with test_utils.concurrent_pipeline_runs_enabled_env():
      with self._mlmd_cm as mlmd_connection_manager:
        m = mlmd_connection_manager.primary_mlmd_handle

        # Sync pipelines with same pipeline_id but different run ids.
        sync_pipelines = [
            _test_pipeline(
                'pipeline1', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run0'
            ),
            _test_pipeline(
                'pipeline1', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run1'
            ),
        ]

        # Should be possible to start the sync pipelines.
        sync_pipeline_states = []
        for pipeline in sync_pipelines:
          sync_pipeline_states.append(
              pipeline_ops.initiate_pipeline_start(m, pipeline)
          )

        async_pipeline = _test_pipeline(
            'pipeline1', pipeline_pb2.Pipeline.ASYNC
        )

        # Starting an async pipeline with the same pipeline_id should be
        # disallowed.
        with self.assertRaises(
            status_lib.StatusNotOkError
        ) as exception_context:
          pipeline_ops.initiate_pipeline_start(m, async_pipeline)
        self.assertEqual(
            status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
        )

        # Deactivate the sync pipelines.
        for pipeline_state in sync_pipeline_states:
          with pipeline_state:
            self.assertTrue(pipeline_state.is_active())
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.COMPLETE
            )

        # Starting async pipeline should be possible now.
        with pipeline_ops.initiate_pipeline_start(
            m, async_pipeline
        ) as pipeline_state:
          self.assertTrue(pipeline_state.is_active())

        # But only once.
        with self.assertRaises(
            status_lib.StatusNotOkError
        ) as exception_context:
          pipeline_ops.initiate_pipeline_start(m, async_pipeline)
        self.assertEqual(
            status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
        )

        # Starting new concurrent runs should be disallowed when an active async
        # pipeline exists.
        new_sync_pipeline = _test_pipeline(
            'pipeline1', pipeline_pb2.Pipeline.SYNC, pipeline_run_id='run2'
        )
        with self.assertRaises(
            status_lib.StatusNotOkError
        ) as exception_context:
          pipeline_ops.initiate_pipeline_start(m, new_sync_pipeline)
        self.assertEqual(
            status_lib.Code.ALREADY_EXISTS, exception_context.exception.code
        )

  def test_check_health_status(self):
    @pipeline_ops._pipeline_op()
    def _fn():
      pass

    # No error should be raised when healthy.
    _fn()

    class _TestEnv(env._DefaultEnv):
      """Unhealthy env for the test."""

      def health_status(self) -> status_lib.Status:
        return status_lib.Status(
            code=status_lib.Code.INTERNAL, message='unhealthy'
        )

    with _TestEnv():
      # Error raised when unhealthy.
      with self.assertRaisesRegex(
          status_lib.StatusNotOkError, 'unhealthy'
      ) as exception_context:
        _fn()
      self.assertEqual(
          status_lib.Code.INTERNAL, exception_context.exception.code
      )

  def test_delete_pipeline_run(self):
    pipeline = test_sync_pipeline.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'test-pipeline-run',
        },
    )

    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      example_gen = pipeline.nodes[0].pipeline_node

      # Initiate a pipeline run.
      pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Fake that the example_gen is RUNNING.
      example_gen_execution = test_utils.fake_example_gen_execution_with_state(
          m,
          example_gen,
          metadata_store_pb2.Execution.State.RUNNING,
      )

      # Fake that the example_gen is COMPLETED with output artifacts.
      contexts = context_lib.prepare_contexts(m, example_gen.contexts)
      execution_publish_utils.publish_succeeded_execution(
          m,
          execution_id=example_gen_execution.id,
          contexts=contexts,
          output_artifacts={'Examples': [standard_artifacts.Examples()]},
      )

      # Check that artifacts have state of LIVE, artifacts path
      # successfully deleted and pipeline execution does not have
      # custom_properties of deleted.
      artifacts = m.store.get_artifacts()
      physical_address = artifacts[0].uri
      self.assertLen(artifacts, 1)
      self.assertEqual(
          artifacts[0].state, metadata_store_pb2.Artifact.State.LIVE
      )
      with pipeline_state:
        self.assertIsNone(
            pipeline_state.execution.custom_properties.get('deleted')
        )

      # Run the function to be tested.
      pipeline_ops.delete_pipeline_run(
          m, pipeline_id='my_pipeline', pipeline_run_id='test-pipeline-run'
      )

      # Make sure that that artifacts have state of DELETED, and pipeline
      # execution has custom_properties of deleted.
      artifacts = m.store.get_artifacts()
      self.assertLen(artifacts, 1)
      self.assertEqual(
          artifacts[0].state, metadata_store_pb2.Artifact.State.DELETED
      )
      self.assertFalse(fileio.exists(physical_address))
      with pipeline_state:
        self.assertTrue(
            pipeline_state.execution.custom_properties.get('deleted')
        )

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  def test_orchestrate_pipelines_with_specified_pipeline_uid(
      self, mock_sync_task_gen
  ):
    with self._mlmd_cm as mlmd_connection_manager:
      m = mlmd_connection_manager.primary_mlmd_handle
      sync_pipelines = [
          _test_pipeline('pipeline1', pipeline_pb2.Pipeline.SYNC),
          _test_pipeline('pipeline2', pipeline_pb2.Pipeline.SYNC),
      ]

      for pipeline in sync_pipelines:
        pipeline_ops.initiate_pipeline_start(m, pipeline)

      # Active executions for active sync pipelines.
      mock_sync_task_gen.return_value.generate.side_effect = [
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[0]
                      ),
                      node_id='Trainer',
                  )
              )
          ],
          [
              test_utils.create_exec_node_task(
                  task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid.from_pipeline(
                          sync_pipelines[1]
                      ),
                      node_id='Trainer',
                  )
              )
          ],
      ]

      task_queue = tq.TaskQueue()
      pipeline_ops.orchestrate(
          mlmd_connection_manager,
          task_queue,
          service_jobs.DummyServiceJobManager(),
          filter_fn=pipeline_ops.filter_by_pipeline_uid(
              task_lib.PipelineUid.from_pipeline_id_and_run_id(
                  pipeline_id='pipeline1', pipeline_run_id='run0'
              )
          ),
      )

      self.assertEqual(1, mock_sync_task_gen.return_value.generate.call_count)

      # Verify there is only one task in the task queue
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertIsInstance(task, task_lib.ExecNodeTask)
      self.assertEqual(
          test_utils.create_node_uid('pipeline1', 'Trainer', 'run0'),
          task.node_uid,
      )
      self.assertTrue(task_queue.is_empty())


if __name__ == '__main__':
  tf.test.main()
