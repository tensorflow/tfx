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
"""Tests for tfx.orchestration.experimental.core.task_scheduler."""

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu


class _FakeTaskScheduler(ts.TaskScheduler):

  def schedule(self):
    return ts.TaskSchedulerResult(
        executor_output=execution_result_pb2.ExecutorOutput())

  def cancel(self):
    pass


class TaskSchedulerRegistryTest(tu.TfxTest):

  def test_registration_and_creation(self):
    # Create a pipeline IR containing deployment config for testing.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['Trainer'].Pack(executor_spec)
    pipeline = pipeline_pb2.Pipeline()
    pipeline.deployment_config.Pack(deployment_config)

    # Register a fake task scheduler.
    spec_type_url = deployment_config.executor_specs['Trainer'].type_url
    ts.TaskSchedulerRegistry.register(spec_type_url, _FakeTaskScheduler)

    # Create a task and verify that the correct scheduler is instantiated.
    task = test_utils.create_exec_node_task(
        node_uid=task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(
                pipeline_id='pipeline', pipeline_run_id=None),
            node_id='Trainer'))
    task_scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
        mock.Mock(), pipeline, task)
    self.assertIsInstance(task_scheduler, _FakeTaskScheduler)


if __name__ == '__main__':
  tf.test.main()
