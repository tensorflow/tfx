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
"""Tests for Subpipeline task scheduler."""

import copy
import os
import threading
import time
import uuid

from absl.testing import flagsaver
from absl.testing import parameterized
from tfx import v1 as tfx
from tfx.dsl.compiler import constants
from tfx.orchestration import data_types_utils
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import subpipeline_task_scheduler
from tfx.orchestration.experimental.core.testing import test_subpipeline
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


class SubpipelineTaskSchedulerTest(test_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._mlmd_cm = mlmd_cm.MLMDConnectionManager.sqlite(metadata_path)
    self.enter_context(self._mlmd_cm)
    self._mlmd_connection = self._mlmd_cm.primary_mlmd_handle

    self._pipeline_run_id = str(uuid.uuid4())
    self._pipeline = self._make_pipeline(pipeline_root, self._pipeline_run_id)

    self._example_gen = test_utils.get_node(self._pipeline, 'my_example_gen')
    self._sub_pipeline = test_utils.get_node(self._pipeline, 'my_sub_pipeline')
    self._transform = test_utils.get_node(self._pipeline, 'my_transform')

    self._task_queue = tq.TaskQueue()

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = test_subpipeline.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_ROOT_PARAMETER_NAME: pipeline_root,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: pipeline_run_id,
        })
    return pipeline

  def test_subpipeline_ir_rewrite(self):
    old_ir = copy.deepcopy(self._sub_pipeline.raw_proto())

    # Asserts original IR is unmodified.
    self.assertProtoEquals(self._sub_pipeline.raw_proto(), old_ir)

  @parameterized.named_parameters(
      dict(testcase_name='run_till_finish', cancel_pipeline=False),
      dict(testcase_name='run_and_cancel', cancel_pipeline=True)
  )
  @flagsaver.flagsaver(subpipeline_scheduler_polling_interval_secs=1.0)
  def test_subpipeline_task_scheduler(self, cancel_pipeline):
    sleep_time = subpipeline_task_scheduler._POLLING_INTERVAL_SECS.value * 5

    with self._mlmd_connection as mlmd_connection:
      test_utils.fake_example_gen_run(mlmd_connection, self._example_gen, 1, 1)

      [sub_pipeline_task] = test_utils.run_generator_and_test(
          test_case=self,
          mlmd_connection_manager=self._mlmd_cm,
          generator_class=sptg.SyncPipelineTaskGenerator,
          pipeline=self._pipeline,
          task_queue=self._task_queue,
          use_task_queue=True,
          service_job_manager=None,
          num_initial_executions=1,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1,
          expected_exec_nodes=[self._sub_pipeline],
          ignore_update_node_state_tasks=True,
          expected_context_names=[
              'my_sub_pipeline', f'my_sub_pipeline_{self._pipeline_run_id}',
              'my_pipeline', self._pipeline_run_id,
              'my_sub_pipeline.my_sub_pipeline'
          ])

      # There should be only 1 orchestrator execution for the outer pipeline.
      pipeline_states = pstate.PipelineState.load_all_active_and_owned(
          mlmd_connection
      )
      self.assertLen(pipeline_states, 1)

      ts_result = []
      scheduler = subpipeline_task_scheduler.SubPipelineTaskScheduler(
          mlmd_handle=mlmd_connection,
          pipeline=self._pipeline,
          task=sub_pipeline_task,
      )

      def start_scheduler(ts_result):
        ts_result.append(scheduler.schedule())
      threading.Thread(target=start_scheduler, args=(ts_result,)).start()

      # Wait for sometime for the update to go through.
      time.sleep(sleep_time)

      # There should be another orchestrator execution for the inner pipeline.
      pipeline_states = pstate.PipelineState.load_all_active_and_owned(
          mlmd_connection
      )
      self.assertLen(pipeline_states, 2)
      sub_pipeline_states = [
          state
          for state in pipeline_states
          if state.pipeline_uid.pipeline_id == 'my_sub_pipeline'
      ]
      self.assertLen(sub_pipeline_states, 1)
      subpipeline_state = pstate.PipelineState.load(
          mlmd_connection,
          sub_pipeline_states[0].pipeline_uid,
      )

      # The scheduler is still waiting for subpipeline to finish.
      self.assertEmpty(ts_result)

      if cancel_pipeline:
        # Call cancel() to initiate the cancel.
        scheduler.cancel(
            task_lib.CancelNodeTask(
                node_uid=task_lib.NodeUid.from_node(
                    self._pipeline,
                    self._sub_pipeline,
                )
            )
        )

        # Sets the cancel state on subpipeline.
        def _cancel(pipeline_state):
          time.sleep(2.0)
          with pipeline_state:
            if pipeline_state.is_stop_initiated():
              pipeline_state.set_pipeline_execution_state(
                  metadata_store_pb2.Execution.CANCELED)
        threading.Thread(target=_cancel, args=(subpipeline_state,)).start()

        # Wait for the update to go through.
        time.sleep(sleep_time)

        self.assertLen(ts_result, 1)
        self.assertEqual(status_lib.Code.CANCELLED, ts_result[0].status.code)
        expected_output_artifacts = {}
      else:
        # directly inject the end node output here...
        expected_output_artifacts = {
            'schema': [tfx.types.standard_artifacts.Schema()]
        }
        end_node = scheduler._sub_pipeline.nodes[-1].pipeline_node
        end_node_execution = execution_lib.prepare_execution(
            mlmd_connection,
            end_node.node_info.type,
            state=metadata_store_pb2.Execution.COMPLETE,
        )
        end_node_contexts = context_lib.prepare_contexts(
            mlmd_connection, end_node.contexts
        )
        execution_lib.put_execution(
            mlmd_connection,
            end_node_execution,
            end_node_contexts,
            output_artifacts=expected_output_artifacts,
            output_event_type=metadata_store_pb2.Event.Type.OUTPUT,
        )
        # Mark inner pipeline as COMPLETE.
        def _complete(pipeline_state):
          with pipeline_state:
            pipeline_state.set_pipeline_execution_state(
                metadata_store_pb2.Execution.COMPLETE)
        threading.Thread(target=_complete, args=(subpipeline_state,)).start()

        # Wait for the update to go through.
        time.sleep(sleep_time)

        self.assertLen(ts_result, 1)
        self.assertEqual(status_lib.Code.OK, ts_result[0].status.code)
        self.assertIsInstance(ts_result[0].output, ts.ExecutorNodeOutput)
      subpipeline_outputs = execution_lib.get_output_artifacts(
          mlmd_connection, sub_pipeline_task.execution_id
      )
      self.assertCountEqual(
          subpipeline_outputs.keys(), expected_output_artifacts.keys()
      )
      for key, values in expected_output_artifacts.items():
        output_artifacts = subpipeline_outputs[key]
        self.assertLen(output_artifacts, 1)
        self.assertLen(values, 1)
        expected_artifact = values[0]
        actual_artifact = output_artifacts[0]
        self.assertEqual(expected_artifact.id, actual_artifact.id)
        self.assertEqual(expected_artifact.type_id, actual_artifact.type_id)

      begin_node_contexts = context_lib.prepare_contexts(
          mlmd_connection,
          scheduler._sub_pipeline.nodes[0].pipeline_node.contexts,
      )
      [begin_node_execution] = (
          execution_lib.get_executions_associated_with_all_contexts(
              mlmd_connection, begin_node_contexts
          )
      )
      self.assertEqual(
          data_types_utils.get_metadata_value(
              begin_node_execution.custom_properties[
                  'injected_begin_node_execution'
              ]
          ),
          'true',
      )
