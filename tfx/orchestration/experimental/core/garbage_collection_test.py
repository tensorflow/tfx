# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.garbage_collection."""

import os
import time
from typing import Iterable, Union

import tensorflow as tf

from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import garbage_collection
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.proto.orchestration import garbage_collection_policy_pb2
from tfx.types.artifact import Artifact
from ml_metadata.proto import metadata_store_pb2


class GarbageCollectionTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = self.create_tempdir()
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._metadata = metadata.Metadata(connection_config=connection_config)
    self._metadata.__enter__()

    pipeline = test_async_pipeline.create_pipeline()
    self._pipeline = pipeline
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node

  def tearDown(self):
    self._metadata.__exit__(None, None, None)
    super().tearDown()

  def assertArtifactIdsEqual(
      self, first: Iterable[Union[metadata_store_pb2.Artifact, Artifact]],
      second: Iterable[Union[metadata_store_pb2.Artifact, Artifact]]) -> None:
    self.assertCountEqual([a.id for a in first], [a.id for a in second])

  def test_no_policy(self):
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0)
    # The examples should not be garbage collected because no garbage collection
    # policy was configured.
    self.assertArtifactIdsEqual(
        [],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid))

  def test_artifacts_in_use(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_most_recently_published=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepMostRecentlyPublished(num_artifacts=0))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0)
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id)
    examples = example_gen_output['examples']
    # The examples should be garbage collected.
    self.assertArtifactIdsEqual(
        examples,
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid))

    test_utils.fake_start_node_with_handle(self._metadata, self._transform,
                                           example_gen_output)
    # The examples should not be garbage collected because they are in use.
    self.assertArtifactIdsEqual(
        [],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid))

  def test_keep_most_recently_published(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_most_recently_published=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepMostRecentlyPublished(num_artifacts=1))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0)
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id)
    examples = example_gen_output['examples']
    # No examples should be garbage collected.
    self.assertArtifactIdsEqual(
        [],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid))

    # Sleep to ensure the second span has a later publish time than the first.
    # The artifact's create_time_since_epoch is set by ML Metadata, and this
    # test uses the ML Metadata C++ Sqlite implementation, so we can't use
    # unittest.mock.patch to change the artifact's create_time_since_epoch.
    time.sleep(1)
    test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=1, version=0)
    # The newest examples should be kept, and the oldest examples should be
    # garbage collected.
    self.assertArtifactIdsEqual(
        examples,
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid))


if __name__ == '__main__':
  tf.test.main()
