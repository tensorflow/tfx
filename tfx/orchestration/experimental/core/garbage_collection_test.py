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
from typing import Iterable

from absl.testing import parameterized
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import garbage_collection
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_async_pipeline

from ml_metadata.proto import metadata_store_pb2


class GarbageCollectionTest(test_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
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
    self._trainer = pipeline.nodes[2].pipeline_node

  def tearDown(self):
    self._metadata.__exit__(None, None, None)
    super().tearDown()

  def _assert_artifacts_equal(self,
                              expected: Iterable[metadata_store_pb2.Artifact],
                              actual) -> None:
    self.assertEqual(set(a.id for a in expected), set(a.id for a in actual))

  def _get_artifacts_to_gc(self):
    return garbage_collection._get_artifacts_to_garbage_collect(
        self._metadata, self._pipeline.pipeline_info.id)

  def test_gc(self):
    example_gen_execution = test_utils.fake_example_gen_run_with_handle(
        self._metadata, self._example_gen, span=1, version=1)
    example_gen_outputs = self._metadata.get_outputs_of_execution(
        example_gen_execution.id)
    example_gen_artifacts = example_gen_outputs['examples']
    # ExampleGen artifact is eligible for garbage collection.
    self._assert_artifacts_equal(example_gen_artifacts,
                                 self._get_artifacts_to_gc())
    transform_execution = test_utils.fake_start_node_with_handle(
        self._metadata, self._transform, example_gen_outputs)
    # ExampleGen artifact is not eligible for garbage collection because
    # they are in use by Transform.
    self._assert_artifacts_equal([], self._get_artifacts_to_gc())
    transform_outputs = test_utils.fake_finish_node_with_handle(
        self._metadata, self._transform, transform_execution.id)
    transform_artifacts = transform_outputs['transform_graph']
    # ExampleGen and Transform artifacts are eligible for garbage collection.
    self._assert_artifacts_equal(example_gen_artifacts + transform_artifacts,
                                 self._get_artifacts_to_gc())


if __name__ == '__main__':
  tf.test.main()
