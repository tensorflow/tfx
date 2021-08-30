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
"""Tests for tfx.orchestration.experimental.core.task_queue."""

import tensorflow as tf
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue
from tfx.orchestration.experimental.core import test_utils
from tfx.utils import test_case_utils as tu


def _test_task(node_id, pipeline_id, key=''):
  node_uid = task_lib.NodeUid(
      pipeline_uid=task_lib.PipelineUid(pipeline_id=pipeline_id, key=key),
      node_id=node_id)
  return test_utils.create_exec_node_task(node_uid)


class TaskQueueTest(tu.TfxTest):

  def test_task_queue_operations(self):
    t1 = _test_task(node_id='trainer', pipeline_id='my_pipeline')
    t2 = _test_task(node_id='transform', pipeline_id='my_pipeline', key='sync')
    tq = task_queue.TaskQueue()

    # Enqueueing new tasks is successful.
    self.assertTrue(tq.enqueue(t1))
    self.assertTrue(tq.enqueue(t2))

    # Re-enqueueing the same tasks fails.
    self.assertFalse(tq.enqueue(t1))
    self.assertFalse(tq.enqueue(t2))

    # Dequeue succeeds and returns `None` when queue is empty.
    self.assertEqual(t1, tq.dequeue())
    self.assertEqual(t2, tq.dequeue())
    self.assertIsNone(tq.dequeue())
    self.assertIsNone(tq.dequeue(0.1))

    # Re-enqueueing the same tasks fails as `task_done` has not been called.
    self.assertFalse(tq.enqueue(t1))
    self.assertFalse(tq.enqueue(t2))

    tq.task_done(t1)
    tq.task_done(t2)

    # Re-enqueueing is allowed after `task_done` has been called.
    self.assertTrue(tq.enqueue(t1))
    self.assertTrue(tq.enqueue(t2))

  def test_invalid_task_done_raises_errors(self):
    t1 = _test_task(node_id='trainer', pipeline_id='my_pipeline')
    t2 = _test_task(node_id='transform', pipeline_id='my_pipeline', key='sync')
    tq = task_queue.TaskQueue()

    # Enqueue t1, but calling `task_done` raises error since t1 is not dequeued.
    self.assertTrue(tq.enqueue(t1))
    with self.assertRaisesRegex(RuntimeError, 'Must call `dequeue`'):
      tq.task_done(t1)

    # `task_done` succeeds after dequeueing.
    self.assertEqual(t1, tq.dequeue())
    tq.task_done(t1)

    # Error since t2 is not in the queue.
    with self.assertRaisesRegex(RuntimeError, 'Task not present'):
      tq.task_done(t2)


if __name__ == '__main__':
  tf.test.main()
