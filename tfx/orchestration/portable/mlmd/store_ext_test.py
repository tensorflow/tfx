# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.mlmd.store_ext."""

import time

import tensorflow as tf
from tfx.orchestration.portable.mlmd import store_ext
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


def _sorted_ids(values):
  return sorted(v.id for v in values)


def _ids(values):
  return [v.id for v in values]


class StoreExtTest(tf.test.TestCase, test_case_utils.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()

  def testGetNodeExecutions(self):
    c = self.put_context('node', 'my-pipeline.my-node')
    e1 = self.put_execution('E', last_known_state='UNKNOWN', contexts=[c])
    e2 = self.put_execution('E', last_known_state='NEW', contexts=[c])
    e3 = self.put_execution('E', last_known_state='RUNNING', contexts=[c])
    e4 = self.put_execution('E', last_known_state='COMPLETE', contexts=[c])
    e5 = self.put_execution('E', last_known_state='FAILED', contexts=[c])
    e6 = self.put_execution('E', last_known_state='CACHED', contexts=[c])
    e7 = self.put_execution('E', last_known_state='CANCELED', contexts=[c])

    with self.subTest('With execution_states unspecified'):
      result = store_ext.get_node_executions(
          self.store, pipeline_id='my-pipeline', node_id='my-node'
      )
      self.assertEqual(_ids(result), _ids([e1, e2, e3, e4, e5, e6, e7]))

    with self.subTest('Bad pipeline_id'):
      result = store_ext.get_node_executions(
          self.store, pipeline_id='not-exist', node_id='my-node'
      )
      self.assertEmpty(result)

    with self.subTest('Bad node_id'):
      result = store_ext.get_node_executions(
          self.store, pipeline_id='my-pipeline', node_id='not-exist'
      )
      self.assertEmpty(result)

    with self.subTest('Multiple execution states'):
      result = store_ext.get_node_executions(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          execution_states=[
              metadata_store_pb2.Execution.RUNNING,
              metadata_store_pb2.Execution.COMPLETE,
              metadata_store_pb2.Execution.FAILED,
              metadata_store_pb2.Execution.CACHED,
              metadata_store_pb2.Execution.CANCELED,
          ],
      )
      self.assertEqual(_ids(result), _ids([e3, e4, e5, e6, e7]))

  def testGetNodeExecutionsCapByArtifactCreateTime(self):
    pipeline_run_context_1 = self.put_context('pipeline_run1', 'run-20230413')
    pipeline_run_context_2 = self.put_context('pipeline_run1', 'run-20230414')
    node_context = self.put_context('node', 'my-pipeline.my-node')
    type_id = self.put_artifact_type('test-artifact-type')
    a1 = metadata_store_pb2.Artifact(type_id=type_id, state='DELETED')
    a2 = metadata_store_pb2.Artifact(
        type_id=type_id, state='MARKED_FOR_DELETION'
    )
    a3 = metadata_store_pb2.Artifact(type_id=type_id, state='LIVE')
    a4 = metadata_store_pb2.Artifact(type_id=type_id, state='LIVE')

    # Insert 3 executions, sleep 1 second in between to ensure executions don't
    # have the same update time.
    self.put_execution(
        'test-execution-type',
        inputs={'input': [a1]},
        outputs={'output': [a2]},
        contexts=[pipeline_run_context_1, node_context],
    )
    time.sleep(1)
    e2 = self.put_execution(
        'test-execution-type',
        inputs={'input': [a2]},
        outputs={'output': [a3]},
        contexts=[pipeline_run_context_2, node_context],
    )
    time.sleep(1)
    e3 = self.put_execution(
        'test-execution-type',
        inputs={'input': [a3]},
        outputs={'output': [a4]},
        contexts=[pipeline_run_context_2, node_context],
    )

    with self.subTest('With execution limited by live artifacts create time'):
      artifacts = store_ext.get_live_output_artifacts_of_node(
          self.store, pipeline_id='my-pipeline', node_id='my-node'
      )
      min_artifact_create_time = min(
          [a.create_time_since_epoch for a in artifacts], default=0
      )
      result = store_ext.get_node_executions(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          min_last_update_time_since_epoch=min_artifact_create_time,
      )
      self.assertEqual(_ids(result), _ids([e2, e3]))

  def testGetLiveOutputArtifactsOfNode(self):
    c = self.put_context('node', 'my-pipeline.my-node')

    with self.subTest('With no LIVE node artifacts'):
      result = store_ext.get_live_output_artifacts_of_node(
          self.store, pipeline_id='my-pipeline', node_id='my-node'
      )
      self.assertEmpty(result)

    x1 = self.put_artifact('X')
    x2 = self.put_artifact('X')
    y1 = self.put_artifact('Y', state='DELETED')
    y2 = self.put_artifact('Y')
    self.put_execution(
        'E', inputs={'x': [x1]}, outputs={'y': [y1]}, contexts=[c]
    )
    self.put_execution(
        'E', inputs={'x': [x2]}, outputs={'y': [y2]}, contexts=[c]
    )

    result = store_ext.get_live_output_artifacts_of_node(
        self.store, pipeline_id='my-pipeline', node_id='my-node'
    )
    self.assertEqual(_sorted_ids(result), _sorted_ids([y2]))

  def testGetLiveOutputArtifactsOfNodeByOutputKeySync(self):
    c1 = self.put_context('pipeline_run', 'run-20230413')
    c2 = self.put_context('node', 'my-pipeline.my-node')

    with self.subTest('With no LIVE node artifacts'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='run-20230413',
      )
      self.assertEmpty(result)

    x1 = self.put_artifact('X')
    x2 = self.put_artifact('X')
    x3 = self.put_artifact('X')
    y1 = self.put_artifact('Y')
    y2 = self.put_artifact('Y', state='DELETED')
    y3 = self.put_artifact('Y')
    y4 = self.put_artifact('Y')
    y5 = self.put_artifact('Y')
    z1 = self.put_artifact('Z')
    z2 = self.put_artifact('Z')
    z3 = self.put_artifact('Z', state='ABANDONED')

    self.put_execution(
        'E',
        inputs={'x': [x1]},
        outputs={'y': [y1], 'z': [z1]},
        contexts=[c1, c2],
    )
    self.put_execution(
        'E',
        inputs={'x': [x2]},
        outputs={'y': [y2, y3, y4], 'z': [z2]},
        contexts=[c1, c2],
    )
    self.put_execution(
        'E',
        inputs={'x': [x3]},
        outputs={'y': [y5], 'z': [z3]},
        contexts=[c1, c2],
    )

    with self.subTest('With execution_states unspecified'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='run-20230413',
      )
      self.assertDictEqual(
          result, {'y': [[y5], [y3, y4], [y1]], 'z': [[], [z2], [z1]]}
      )

    with self.subTest('With execution_states=[COMPLETE, CACHED]'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='',
          execution_states=[
              metadata_store_pb2.Execution.COMPLETE,
              metadata_store_pb2.Execution.CACHED,
          ],
      )
      self.assertDictEqual(
          result,
          {'y': [[y5], [y3, y4], [y1]], 'z': [[], [z2], [z1]]},
      )

  def testGetLiveOutputArtifactsOfNodeByOutputKeyAsync(self):
    c1 = self.put_context('node', 'my-pipeline.my-node')

    with self.subTest('With no LIVE node artifacts'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='',
      )
      self.assertEmpty(result)

    x1 = self.put_artifact('X')
    x2 = self.put_artifact('X')
    x3 = self.put_artifact('X')
    # Intermediate artifact
    x4 = self.put_artifact('X')

    y1 = self.put_artifact('Y')
    y2 = self.put_artifact('Y', state='DELETED')
    y3 = self.put_artifact('Y')
    y4 = self.put_artifact('Y')
    y5 = self.put_artifact('Y')
    # Intermediate artifact
    y6 = self.put_artifact('Y')

    z1 = self.put_artifact('Z')
    z2 = self.put_artifact('Z')
    z3 = self.put_artifact('Z', state='ABANDONED')
    # Intermediate artifact
    z4 = self.put_artifact('Z')

    self.put_execution(
        'E',
        last_known_state='CACHED',
        inputs={'x': [x1]},
        outputs={'y': [y1], 'z': [z1]},
        contexts=[c1],
    )
    self.put_execution(
        'E',
        last_known_state='COMPLETE',
        inputs={'x': [x2]},
        outputs={'y': [y2, y3, y4], 'z': [z2]},
        contexts=[c1],
    )
    self.put_execution(
        'E',
        last_known_state='COMPLETE',
        inputs={'x': [x3]},
        outputs={'y': [y5], 'z': [z3]},
        contexts=[c1],
    )
    # A special case for intermediate artifacts, which can be LIVE even the
    # execution is FAILED.
    self.put_execution(
        'E',
        last_known_state='FAILED',
        inputs={'x': [x4]},
        outputs={'y': [y6], 'z': [z4]},
        contexts=[c1],
    )

    with self.subTest('With execution_states unspecified'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='',
      )
      self.assertDictEqual(
          result,
          {'y': [[y6], [y5], [y3, y4], [y1]], 'z': [[z4], [], [z2], [z1]]},
      )

    with self.subTest('With execution_states=[COMPLETE, CACHED, FAILED]'):
      result = store_ext.get_live_output_artifacts_of_node_by_output_key(
          self.store,
          pipeline_id='my-pipeline',
          node_id='my-node',
          pipeline_run_id='',
          execution_states=[
              metadata_store_pb2.Execution.COMPLETE,
              metadata_store_pb2.Execution.CACHED,
              metadata_store_pb2.Execution.FAILED,
          ],
      )
      self.assertDictEqual(
          result,
          {'y': [[y6], [y5], [y3, y4], [y1]], 'z': [[z4], [], [z2], [z1]]},
      )
