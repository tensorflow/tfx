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

import tensorflow as tf
from tfx.orchestration.portable.mlmd import store_ext
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


def _ids(values):
  return [v.id for v in values]


class StoreExtTest(tf.test.TestCase, test_case_utils.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()

  def testGetSuccessfulNodeExecutions(self):
    c = self.put_context('node', 'my-pipeline.my-node')
    self.put_execution('E', last_known_state='UNKNOWN', contexts=[c])
    self.put_execution('E', last_known_state='NEW', contexts=[c])
    self.put_execution('E', last_known_state='RUNNING', contexts=[c])
    e1 = self.put_execution('E', last_known_state='COMPLETE', contexts=[c])
    self.put_execution('E', last_known_state='FAILED', contexts=[c])
    e2 = self.put_execution('E', last_known_state='CACHED', contexts=[c])
    self.put_execution('E', last_known_state='CANCELED', contexts=[c])

    result = store_ext.get_successful_node_executions(
        self.store, pipeline_id='my-pipeline', node_id='my-node'
    )
    self.assertCountEqual(_ids(result), _ids([e1, e2]))

    with self.subTest('Bad pipeline_id'):
      result = store_ext.get_successful_node_executions(
          self.store, pipeline_id='not-exist', node_id='my-node'
      )
      self.assertEmpty(result)

    with self.subTest('Bad node_id'):
      result = store_ext.get_successful_node_executions(
          self.store, pipeline_id='my-pipeline', node_id='not-exist'
      )
      self.assertEmpty(result)

  def testGetOutputArtifactsFromExecutionIds(self):
    x1 = self.put_artifact('X')
    x2 = self.put_artifact('X')
    y1 = self.put_artifact('Y', state='DELETED')
    y2 = self.put_artifact('Y')
    y3 = self.put_artifact('Y')
    y4 = self.put_artifact('Y')
    e1 = self.put_execution('E', inputs={'x': [x1]}, outputs={'y': [y1]})
    e2 = self.put_execution('E', inputs={'x': [x2]}, outputs={'y': [y2]})
    e3 = self.put_execution(
        'E',
        inputs={'x': [x2]},
        outputs={'y': [y3]},
        output_event_type=metadata_store_pb2.Event.DECLARED_OUTPUT,
    )
    e4 = self.put_execution(
        'E',
        inputs={'x': [x2]},
        outputs={'y': [y4]},
        output_event_type=metadata_store_pb2.Event.INTERNAL_OUTPUT,
    )

    result = store_ext.get_output_artifacts_from_execution_ids(
        self.store,
        execution_ids=_ids([e1, e2, e3, e4]),
    )
    self.assertCountEqual(_ids(result), _ids([y1, y2, y3, y4]))

    with self.subTest('With artifact filter'):
      result = store_ext.get_output_artifacts_from_execution_ids(
          self.store,
          execution_ids=_ids([e1, e2, e3, e4]),
          artifact_filter='state = LIVE',
      )
      self.assertCountEqual(_ids(result), _ids([y2, y3, y4]))

  def testGetLiveOutputArtifactsOfNode(self):
    c = self.put_context('node', 'my-pipeline.my-node')
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
    self.assertCountEqual(_ids(result), _ids([y2]))


if __name__ == '__main__':
  tf.test.main()
