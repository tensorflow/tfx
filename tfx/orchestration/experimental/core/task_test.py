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
"""Tests for tfx.orchestration.experimental.core.task."""

import tensorflow as tf
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu


class TaskTest(tu.TfxTest):

  def test_node_uid_from_node(self):
    pipeline = pipeline_pb2.Pipeline()
    pipeline.pipeline_info.id = 'pipeline'
    node = pipeline_pb2.PipelineNode()
    node.node_info.id = 'Trainer'
    self.assertEqual(
        task_lib.NodeUid(
            pipeline_uid=task_lib.PipelineUid(pipeline_id='pipeline'),
            node_id='Trainer'),
        task_lib.NodeUid.from_node(pipeline, node))

  def test_task_type_ids(self):
    self.assertEqual('ExecNodeTask', task_lib.ExecNodeTask.task_type_id())
    self.assertEqual('CancelNodeTask', task_lib.CancelNodeTask.task_type_id())

  def test_task_ids(self):
    pipeline_uid = task_lib.PipelineUid(pipeline_id='pipeline')
    node_uid = task_lib.NodeUid(pipeline_uid=pipeline_uid, node_id='Trainer')
    exec_node_task = test_utils.create_exec_node_task(node_uid)
    self.assertEqual(('ExecNodeTask', node_uid), exec_node_task.task_id)
    cancel_node_task = task_lib.CancelNodeTask(node_uid=node_uid)
    self.assertEqual(('CancelNodeTask', node_uid), cancel_node_task.task_id)


if __name__ == '__main__':
  tf.test.main()
