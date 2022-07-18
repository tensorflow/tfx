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
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu


class _FakeTaskScheduler(ts.TaskScheduler):

  def schedule(self):
    return ts.TaskSchedulerResult(
        output=ts.ExecutorNodeOutput(
            executor_output=execution_result_pb2.ExecutorOutput()))

  def cancel(self):
    pass


def _fake_task_scheduler_builder(mlmd_handle: metadata.Metadata,
                                 pipeline: pipeline_pb2.Pipeline,
                                 task: task_lib.Task) -> ts.TaskScheduler:
  return _FakeTaskScheduler(mlmd_handle, pipeline, task)


class TaskSchedulerRegistryTest(tu.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline = pipeline_pb2.Pipeline()
    pipeline.pipeline_info.id = 'pipeline'
    pipeline.nodes.add().pipeline_node.node_info.id = 'Trainer'
    pipeline.nodes.add().pipeline_node.node_info.id = 'Transform'
    importer_node = pipeline.nodes.add().pipeline_node
    importer_node.node_info.id = 'Importer'
    importer_node.node_info.type.name = constants.IMPORTER_NODE_TYPE
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['Trainer'].Pack(executor_spec)
    pipeline.deployment_config.Pack(deployment_config)
    self._spec_type_url = deployment_config.executor_specs['Trainer'].type_url
    self._pipeline = pipeline
    ts.TaskSchedulerRegistry.clear()

  def test_register_using_executor_spec_type_url(self):
    # Register a fake task scheduler.
    ts.TaskSchedulerRegistry.register(self._spec_type_url, _FakeTaskScheduler)

    # Create a task and verify that the correct scheduler is instantiated.
    task = test_utils.create_exec_node_task(
        node_uid=task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(pipeline_id='pipeline'),
            node_id='Trainer'),
        pipeline=self._pipeline)
    task_scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
        mock.Mock(), self._pipeline, task)
    self.assertIsInstance(task_scheduler, _FakeTaskScheduler)

  def test_register_using_node_type_name(self):
    # Register a fake task scheduler.
    ts.TaskSchedulerRegistry.register(constants.IMPORTER_NODE_TYPE,
                                      _FakeTaskScheduler)

    # Create a task and verify that the correct scheduler is instantiated.
    task = test_utils.create_exec_node_task(
        node_uid=task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(pipeline_id='pipeline'),
            node_id='Importer'),
        pipeline=self._pipeline)
    task_scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
        mock.Mock(), self._pipeline, task)
    self.assertIsInstance(task_scheduler, _FakeTaskScheduler)

  def test_register_using_builder_function(self):
    # Register a fake task scheduler builder.
    ts.TaskSchedulerRegistry.register(self._spec_type_url,
                                      _fake_task_scheduler_builder)

    # Create a task and verify that the correct scheduler is instantiated.
    task = test_utils.create_exec_node_task(
        node_uid=task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(pipeline_id='pipeline'),
            node_id='Trainer'),
        pipeline=self._pipeline)
    task_scheduler = ts.TaskSchedulerRegistry.create_task_scheduler(
        mock.Mock(), self._pipeline, task)
    self.assertIsInstance(task_scheduler, _FakeTaskScheduler)

  def test_scheduler_not_found(self):
    task = test_utils.create_exec_node_task(
        node_uid=task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(pipeline_id='pipeline'),
            node_id='Transform'),
        pipeline=self._pipeline)
    with self.assertRaisesRegex(ValueError,
                                'No task scheduler class or builder found'):
      ts.TaskSchedulerRegistry.create_task_scheduler(mock.Mock(),
                                                     self._pipeline, task)


if __name__ == '__main__':
  tf.test.main()
