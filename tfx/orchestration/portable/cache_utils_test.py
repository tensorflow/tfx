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
"""Tests for tfx.orchestration.portable.cache_utils."""
import os
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.portable import cache_utils
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils
from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class CacheUtilsTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._module_file_path = os.path.join(self.tmp_dir, 'module_file')
    self._input_artifacts = {'input_examples': [standard_artifacts.Examples()]}
    self._output_artifacts = {'output_models': [standard_artifacts.Model()]}
    self._parameters = {'module_file': self._module_file_path}
    self._module_file_content = 'module content'
    self._pipeline_node = text_format.Parse(
        """
        node_info {
          id: "my_id"
        }
        """, pipeline_pb2.PipelineNode())
    self._pipeline_info = pipeline_pb2.PipelineInfo(id='pipeline_id')
    self._executor_spec = text_format.Parse(
        """
        class_path: "my.class.path"
        """, executable_spec_pb2.PythonClassExecutableSpec())

  def _get_cache_context(self,
                         metadata_handler,
                         custom_pipeline_node=None,
                         custom_pipeline_info=None,
                         executor_spec=None,
                         custom_input_artifacts=None,
                         custom_output_artifacts=None,
                         custom_parameters=None,
                         custom_module_content=None):
    with fileio.open(self._module_file_path, 'w+') as f:
      f.write(custom_module_content or self._module_file_content)
    return cache_utils.get_cache_context(
        metadata_handler,
        custom_pipeline_node or self._pipeline_node,
        custom_pipeline_info or self._pipeline_info,
        executor_spec=(executor_spec or self._executor_spec),
        input_artifacts=(custom_input_artifacts or self._input_artifacts),
        output_artifacts=(custom_output_artifacts or self._output_artifacts),
        parameters=(custom_parameters or self._parameters))

  def testGetCacheContext(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      cache_context = self._get_cache_context(m)
      [context_from_mlmd] = m.store.get_contexts()
      self.assertProtoPartiallyEquals(
          cache_context,
          context_from_mlmd,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

  def testGetCacheContextTwiceSameArgs(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(m)
      # Same args should not create a new cache context.
      self.assertLen(m.store.get_contexts(), 1)

  def testGetCacheContextTwiceDifferentOutputUri(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      output_model_different_uri = standard_artifacts.Model()
      output_model_different_uri.uri = 'diff_uri'
      self._get_cache_context(
          m,
          custom_output_artifacts={
              'output_models': [output_model_different_uri]
          })
      # Only different output uri should not create a new cache context.
      self.assertLen(m.store.get_contexts(), 1)

  def testGetCacheContextTwiceDifferentOutputs(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(
          m, custom_output_artifacts={'k': [standard_artifacts.Model()]})
      # Different output skeleton will result in a new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentInputs(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(
          m, custom_input_artifacts={'k': [standard_artifacts.Examples(),]})
      # Different input artifacts will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentParameters(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(m, custom_parameters={'new_prop': 'value'})
      # Different parameters will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentModuleContent(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(m, custom_module_content='new module content')
      # Different module file content will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentPipelineInfo(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(
          m, custom_pipeline_info=pipeline_pb2.PipelineInfo(id='new_id'))
      # Different pipeline info will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentNodeInfo(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(
          m,
          custom_pipeline_node=text_format.Parse(
              """
              node_info {
                id: "new_node_id"
              }
              """, pipeline_pb2.PipelineNode()))
      # Different executor spec will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCacheContextTwiceDifferentExecutorSpec(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self._get_cache_context(m)
      self._get_cache_context(
          m,
          executor_spec=text_format.Parse(
              """
              class_path: "new.class.path"
              """, executable_spec_pb2.PythonClassExecutableSpec()))
      # Different executor spec will result in new cache context.
      self.assertLen(m.store.get_contexts(), 2)

  def testGetCachedOutputArtifacts(self):
    # Output artifacts that will be used by the first execution with the same
    # cache key.
    output_model_one = standard_artifacts.Model()
    output_model_one.uri = 'model_one'
    output_model_two = standard_artifacts.Model()
    output_model_two.uri = 'model_two'
    output_example_one = standard_artifacts.Examples()
    output_example_one.uri = 'example_one'
    # Output artifacts that will be used by the second execution with the same
    # cache key.
    output_model_three = standard_artifacts.Model()
    output_model_three.uri = 'model_three'
    output_model_four = standard_artifacts.Model()
    output_model_four.uri = 'model_four'
    output_example_two = standard_artifacts.Examples()
    output_example_two.uri = 'example_two'
    output_models_key = 'output_models'
    output_examples_key = 'output_examples'
    with metadata.Metadata(connection_config=self._connection_config) as m:
      cache_context = context_lib.register_context_if_not_exists(
          m, context_lib.CONTEXT_TYPE_EXECUTION_CACHE, 'cache_key')
      cached_output = cache_utils.get_cached_outputs(m, cache_context)
      # No succeed execution is associate with this context yet, so the cached
      # output is None
      self.assertIsNone(cached_output)
      execution_one = execution_publish_utils.register_execution(
          m, metadata_store_pb2.ExecutionType(name='my_type'), [cache_context])
      execution_publish_utils.publish_succeeded_execution(
          m,
          execution_one.id, [cache_context],
          output_artifacts={
              output_models_key: [output_model_one, output_model_two],
              output_examples_key: [output_example_one]
          })
      execution_two = execution_publish_utils.register_execution(
          m, metadata_store_pb2.ExecutionType(name='my_type'), [cache_context])
      output_artifacts = execution_publish_utils.publish_succeeded_execution(
          m,
          execution_two.id, [cache_context],
          output_artifacts={
              output_models_key: [output_model_three, output_model_four],
              output_examples_key: [output_example_two]
          })
      # The cached output got should be the artifacts produced by the most
      # recent execution under the given cache context.
      cached_output = cache_utils.get_cached_outputs(m, cache_context)
      self.assertLen(cached_output, 2)
      self.assertLen(cached_output[output_models_key], 2)
      self.assertLen(cached_output[output_examples_key], 1)
      self.assertProtoPartiallyEquals(
          cached_output[output_models_key][0].mlmd_artifact,
          output_artifacts[output_models_key][0].mlmd_artifact,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      self.assertProtoPartiallyEquals(
          cached_output[output_models_key][1].mlmd_artifact,
          output_artifacts[output_models_key][1].mlmd_artifact,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      self.assertProtoPartiallyEquals(
          cached_output[output_examples_key][0].mlmd_artifact,
          output_artifacts[output_examples_key][0].mlmd_artifact,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

  def testGetCachedOutputArtifactsForNodesWithNoOuput(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      cache_context = context_lib.register_context_if_not_exists(
          m, context_lib.CONTEXT_TYPE_EXECUTION_CACHE, 'cache_key')
      cached_output = cache_utils.get_cached_outputs(m, cache_context)
      # No succeed execution is associate with this context yet, so the cached
      # output is None.
      self.assertIsNone(cached_output)
      execution_one = execution_publish_utils.register_execution(
          m, metadata_store_pb2.ExecutionType(name='my_type'), [cache_context])
      execution_publish_utils.publish_succeeded_execution(
          m,
          execution_one.id, [cache_context])
      cached_output = cache_utils.get_cached_outputs(m, cache_context)
      # A succeed execution is associate with this context, so the cached
      # output is not None but an empty dict.
      self.assertIsNotNone(cached_output)
      self.assertEmpty(cached_output)


if __name__ == '__main__':
  tf.test.main()
