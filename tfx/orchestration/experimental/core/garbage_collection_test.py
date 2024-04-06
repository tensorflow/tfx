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
from typing import Iterable, Optional, Union

from absl import logging
from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import garbage_collection
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.proto.orchestration import garbage_collection_policy_pb2
from tfx.types.artifact import Artifact

from ml_metadata.proto import metadata_store_pb2


class GarbageCollectionTest(test_utils.TfxTest, parameterized.TestCase):

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

  def _produce_examples(
      self,
      span: Optional[int] = 0,
      version: Optional[int] = 0,
      **additional_custom_properties) -> Artifact:
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span, version,
        **additional_custom_properties)
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id)
    return example_gen_output['examples'][0]

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
            self._metadata, example_gen_node_uid, self._example_gen))

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
            self._metadata, example_gen_node_uid, self._example_gen))

    test_utils.fake_start_node_with_handle(self._metadata, self._transform,
                                           example_gen_output)
    # The examples should not be garbage collected because they are in use.
    self.assertArtifactIdsEqual(
        [],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen))

  def test_artifacts_external(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_most_recently_published=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepMostRecentlyPublished(num_artifacts=0))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    expected_to_be_garbage_collected = self._produce_examples(is_external=True)
    # The example should not be garbage collected because it is external.
    self.assertArtifactIdsEqual(
        [expected_to_be_garbage_collected],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen
        ),
    )

  def test_artifacts_external_counted_for_policy(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_most_recently_published=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepMostRecentlyPublished(num_artifacts=1))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)

    expected_to_be_garbage_collected = self._produce_examples(is_external=True)
    self._produce_examples(
        is_external=True
    )  # Most recent one should not be garbage collected.
    self.assertArtifactIdsEqual(
        [expected_to_be_garbage_collected],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen))

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
            self._metadata, example_gen_node_uid, self._example_gen))

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
            self._metadata, example_gen_node_uid, self._example_gen))

  @mock.patch.object(fileio, 'remove')
  def test_garbage_collect_artifacts(self, remove):
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0)
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id)
    examples = example_gen_output['examples']
    examples_protos = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples])

    garbage_collection.garbage_collect_artifacts(self._metadata,
                                                 examples_protos)

    remove.assert_called_once_with(examples[0].uri)
    self.assertEqual(
        metadata_store_pb2.Artifact.State.DELETED,
        self._metadata.store.get_artifacts_by_id([examples[0].id])[0].state,
    )

  @mock.patch.object(garbage_collection, '_delete_artifact_uri', autospec=True)
  def test_garbage_collect_external_artifacts(self, mock_delete_artifact_uri):
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0, is_external=True
    )
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id
    )
    examples = example_gen_output['examples']
    examples_protos = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )

    garbage_collection.garbage_collect_artifacts(
        self._metadata, examples_protos
    )

    mock_delete_artifact_uri.assert_not_called()
    self.assertEqual(
        metadata_store_pb2.Artifact.State.DELETED,
        self._metadata.store.get_artifacts_by_id([examples[0].id])[0].state,
    )

  @mock.patch.object(fileio, 'remove')
  def test_garbage_collect_artifacts_output_of_failed_executions(self, remove):
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0
    )
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id
    )
    examples = example_gen_output['examples']
    examples_protos = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )
    example_gen_execution.last_known_state = metadata_store_pb2.Execution.FAILED
    self._metadata.store.put_execution(
        example_gen_execution, artifact_and_events=[], contexts=[]
    )
    garbage_collection.garbage_collect_artifacts(
        self._metadata, examples_protos
    )

    remove.assert_called_once_with(examples[0].uri)
    self.assertEqual(
        metadata_store_pb2.Artifact.State.DELETED,
        self._metadata.store.get_artifacts_by_id([examples[0].id])[0].state,
    )

  @mock.patch.object(fileio, 'exists')
  def test_garbage_collect_artifacts_does_not_throw_and_marks_deleted_when_not_found(
      self, mock_exists
  ):
    mock_exists.return_value = False
    test_dir = self.create_tempdir()
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0
    )
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id
    )
    examples = example_gen_output['examples']
    examples_protos = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )
    for examples_proto in examples_protos:
      examples_proto.uri = os.path.join(test_dir, 'does/not/exist')

    garbage_collection.garbage_collect_artifacts(
        self._metadata, examples_protos
    )

    mock_exists.assert_called_once()

    # Also make sure the artifacts are still marked as DELETED.
    final_artifacts = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )
    for artifact in final_artifacts:
      with self.subTest():
        self.assertEqual(artifact.state, metadata_store_pb2.Artifact.DELETED)

  @mock.patch.object(fileio, 'remove')
  @mock.patch.object(fileio, 'exists')
  def test_garbage_collect_artifacts_does_not_throw_or_mark_deleted_when_permission_denied(
      self, mock_exists, mock_remove
  ):
    mock_exists.return_value = True
    mock_remove.side_effect = PermissionError('permission denied')
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=0, version=0
    )
    example_gen_output = self._metadata.get_outputs_of_execution(
        example_gen_execution.id
    )
    examples = example_gen_output['examples']
    examples_protos = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )

    garbage_collection.garbage_collect_artifacts(
        self._metadata, examples_protos
    )

    # Also make sure the artifacts are not marked as DELETED.
    final_artifacts = self._metadata.store.get_artifacts_by_id(
        [e.id for e in examples]
    )
    for artifact in final_artifacts:
      with self.subTest():
        self.assertNotEqual(artifact.state, metadata_store_pb2.Artifact.DELETED)

  @mock.patch.object(garbage_collection, 'garbage_collect_artifacts')
  @mock.patch.object(logging, 'exception')
  def test_run_garbage_collect_for_node_catches_garbage_collect_artifacts_error(
      self,
      logging_exception,
      garbage_collect_artifacts,
  ):
    garbage_collect_artifacts.side_effect = Exception('Failed!')
    example_gen_node_uid = task_lib.NodeUid.from_node(
        self._pipeline, self._example_gen
    )
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    try:
      garbage_collection.run_garbage_collection_for_node(
          self._metadata, example_gen_node_uid, self._example_gen
      )
    except:  # pylint: disable=bare-except
      self.fail('Error was raised')
    logs = logging_exception.call_args_list
    self.assertLen(logs, 1)
    self.assertStartsWith(logs[0].args[0], r'Garbage collection for node')

  @mock.patch.object(
      garbage_collection, 'get_artifacts_to_garbage_collect_for_node'
  )
  @mock.patch.object(logging, 'exception')
  def test_run_garbage_collect_for_node_catches_get_artifacts_to_garbage_collect_for_node_error(
      self, logging_exception, get_artifacts_to_garbage_collect_for_node
  ):
    get_artifacts_to_garbage_collect_for_node.side_effect = Exception('Failed!')
    example_gen_node_uid = task_lib.NodeUid.from_node(
        self._pipeline, self._example_gen
    )
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)
    try:
      garbage_collection.run_garbage_collection_for_node(
          self._metadata, example_gen_node_uid, self._example_gen
      )
    except:  # pylint: disable=bare-except
      self.fail('Error was raised')
    logs = logging_exception.call_args_list
    self.assertLen(logs, 1)
    self.assertStartsWith(logs[0].args[0], r'Garbage collection for node')

  def test_keep_property_value_groups(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_property_value_groups=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepPropertyValueGroups(groupings=[
            garbage_collection_policy_pb2.GarbageCollectionPolicy
            .KeepPropertyValueGroups.Grouping(
                property_name='examples_type.name'),
            garbage_collection_policy_pb2.GarbageCollectionPolicy
            .KeepPropertyValueGroups.Grouping(
                property_name='span',
                keep_num=2,
                keep_order=garbage_collection_policy_pb2.GarbageCollectionPolicy
                .KeepPropertyValueGroups.Grouping.KeepOrder.KEEP_ORDER_LARGEST),
            garbage_collection_policy_pb2.GarbageCollectionPolicy
            .KeepPropertyValueGroups.Grouping(
                property_name='version',
                keep_num=1,
                keep_order=garbage_collection_policy_pb2.GarbageCollectionPolicy
                .KeepPropertyValueGroups.Grouping.KeepOrder.KEEP_ORDER_LARGEST)
        ]))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)

    examples_a_0_0 = self._produce_examples(0, 0)
    examples_a_1_0 = self._produce_examples(1, 0)
    examples_a_2_0 = self._produce_examples(2, 0)
    self._produce_examples(2, 1)  # Should not be garbage collected
    self._produce_examples(3, 0)  # Should not be garbage collected
    self.assertArtifactIdsEqual(
        [examples_a_0_0, examples_a_1_0, examples_a_2_0],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen))

  def test_keep_property_value_groups_with_none_value(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_property_value_groups=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepPropertyValueGroups(groupings=[
            garbage_collection_policy_pb2.GarbageCollectionPolicy
            .KeepPropertyValueGroups.Grouping(
                property_name='test_property',
                keep_num=2,
                keep_order=garbage_collection_policy_pb2.GarbageCollectionPolicy
                .KeepPropertyValueGroups.Grouping.KeepOrder.KEEP_ORDER_SMALLEST)
        ]))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)

    self._produce_examples(test_property=1)  # Should not be garbage collected
    examples_none = self._produce_examples()  #  Should not be garbage collected
    self.assertArtifactIdsEqual(
        [],
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen))

    self._produce_examples(test_property=2)  # Should not be garbage collected
    self.assertArtifactIdsEqual(
        [examples_none],  # Now it should be garbage collected
        garbage_collection.get_artifacts_to_garbage_collect_for_node(
            self._metadata, example_gen_node_uid, self._example_gen))

  def test_keep_property_value_groups_non_homogenous_types_failure(self):
    policy = garbage_collection_policy_pb2.GarbageCollectionPolicy(
        keep_property_value_groups=garbage_collection_policy_pb2
        .GarbageCollectionPolicy.KeepPropertyValueGroups(groupings=[
            garbage_collection_policy_pb2.GarbageCollectionPolicy
            .KeepPropertyValueGroups.Grouping(property_name='test_property')
        ]))
    self._example_gen.outputs.outputs[
        'examples'].garbage_collection_policy.CopyFrom(policy)
    example_gen_node_uid = task_lib.NodeUid.from_node(self._pipeline,
                                                      self._example_gen)
    pipeline_ops.initiate_pipeline_start(self._metadata, self._pipeline)

    self._produce_examples(test_property=0)
    self._produce_examples(test_property='str')

    expected_error_message = (
        'Properties from the same group should have a homogenous type except '
        'NoneType. Expected <class \'%s\'>, but passed <class \'%s\'>')
    # Embrace all order cases.
    with self.assertRaisesRegex(
        ValueError, (f'({expected_error_message % ("str", "int")}|'
                     f'{expected_error_message % ("int", "str")})')):
      garbage_collection.get_artifacts_to_garbage_collect_for_node(
          self._metadata, example_gen_node_uid, self._example_gen)


if __name__ == '__main__':
  tf.test.main()
