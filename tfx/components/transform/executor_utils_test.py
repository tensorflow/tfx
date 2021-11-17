# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.transform.executor_utils."""

import tensorflow as tf
from tfx.components.transform import executor_utils
from tfx.components.transform import labels
from tfx.proto import transform_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils


class ExecutorUtilsTest(tf.test.TestCase):

  def testMaybeBindCustomConfig(self):

    def dummy(custom_config):
      return custom_config

    patched = executor_utils.MaybeBindCustomConfig(
        {labels.CUSTOM_CONFIG: '{"value":42}'}, dummy)
    self.assertEqual({'value': 42}, patched())

  def testValidateOnlyOneSpecified(self):
    executor_utils.ValidateOnlyOneSpecified({'a': 1}, ('a', 'b', 'c'))

    with self.assertRaisesRegex(ValueError, 'One of'):
      executor_utils.ValidateOnlyOneSpecified({'z': 1}, ('a', 'b', 'c'))

    with self.assertRaisesRegex(ValueError, 'At most one of'):
      executor_utils.ValidateOnlyOneSpecified({
          'a': [1],
          'b': '1'
      }, ('a', 'b', 'c'))

  def testValidateOnlyOneSpecifiedAllowMissing(self):
    executor_utils.ValidateOnlyOneSpecified({'z': 1}, ('a', 'b', 'c'), True)

    with self.assertRaisesRegex(ValueError, 'At most one of'):
      executor_utils.ValidateOnlyOneSpecified({
          'a': [1],
          'b': '1'
      }, ('a', 'b', 'c'), True)

  def testMatchNumberOfTransformedExamplesArtifacts(self):
    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [
            standard_artifacts.Examples(),
            standard_artifacts.Examples()
        ]
    }
    original_output_artifact = standard_artifacts.Examples()
    original_output_artifact.uri = '/dummy/path'
    output_dict = {
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY: [
            original_output_artifact
        ]
    }
    executor_utils.MatchNumberOfTransformedExamplesArtifacts(
        input_dict, output_dict)
    self.assertLen(
        output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY], 2)
    # Uris of the new artifacts should be located under the original artifact.
    self.assertTrue(output_dict[
        standard_component_specs.TRANSFORMED_EXAMPLES_KEY][0].uri.startswith(
            original_output_artifact.uri))

  def testResolveSplitsConfigEmptyAnalyze(self):
    wrong_config = transform_pb2.SplitsConfig(transform=['train'])
    with self.assertRaisesRegex(ValueError, 'analyze cannot be empty'):
      config_str = proto_utils.proto_to_json(wrong_config)
      executor_utils.ResolveSplitsConfig(config_str, [])

  def testResolveSplitsConfigOk(self):
    config = transform_pb2.SplitsConfig(
        analyze=['train'], transform=['train', 'eval'])
    config_str = proto_utils.proto_to_json(config)
    resolved = executor_utils.ResolveSplitsConfig(config_str, [])
    self.assertProtoEquals(config, resolved)

  def testResolveSplitsConfigInconsistentSplits(self):
    examples1 = standard_artifacts.Examples()
    examples1.split_names = artifact_utils.encode_split_names(['train'])
    examples2 = standard_artifacts.Examples()
    examples2.split_names = artifact_utils.encode_split_names(['train', 'test'])
    with self.assertRaisesRegex(ValueError, 'same split names'):
      executor_utils.ResolveSplitsConfig(None, [examples1, examples2])

  def testResolveSplitsConfigDefault(self):
    examples1 = standard_artifacts.Examples()
    examples1.split_names = artifact_utils.encode_split_names(['train', 'test'])
    examples2 = standard_artifacts.Examples()
    examples2.split_names = artifact_utils.encode_split_names(['train', 'test'])

    resolved = executor_utils.ResolveSplitsConfig(None, [examples1, examples2])
    self.assertEqual(set(resolved.analyze), {'train'})
    self.assertEqual(set(resolved.transform), {'train', 'test'})

  def testSetSplitNames(self):
    # Should work with None.
    executor_utils.SetSplitNames(['train'], None)

    examples1 = standard_artifacts.Examples()
    examples2 = standard_artifacts.Examples()
    executor_utils.SetSplitNames(['train'], [examples1, examples2])

    self.assertEqual(examples1.split_names, '["train"]')
    self.assertEqual(examples2.split_names, examples1.split_names)

  def testGetSplitPaths(self):
    # Should work with None.
    self.assertEmpty(executor_utils.GetSplitPaths(None))

    examples1 = standard_artifacts.Examples()
    examples1.uri = '/uri1'
    examples2 = standard_artifacts.Examples()
    examples2.uri = '/uri2'
    executor_utils.SetSplitNames(['train', 'test'], [examples1, examples2])

    paths = executor_utils.GetSplitPaths([examples1, examples2])
    self.assertCountEqual([
        '/uri1/Split-train/transformed_examples',
        '/uri2/Split-train/transformed_examples',
        '/uri1/Split-test/transformed_examples',
        '/uri2/Split-test/transformed_examples'
    ], paths)

  def testGetCachePathEntry(self):
    # Empty case.
    self.assertEmpty(
        executor_utils.GetCachePathEntry(
            standard_component_specs.ANALYZER_CACHE_KEY, {}))

    cache_artifact = standard_artifacts.TransformCache()
    cache_artifact.uri = '/dummy'
    # input
    result = executor_utils.GetCachePathEntry(
        standard_component_specs.ANALYZER_CACHE_KEY,
        {standard_component_specs.ANALYZER_CACHE_KEY: [cache_artifact]})
    self.assertEqual({labels.CACHE_INPUT_PATH_LABEL: '/dummy'}, result)

    # output
    result = executor_utils.GetCachePathEntry(
        standard_component_specs.UPDATED_ANALYZER_CACHE_KEY,
        {standard_component_specs.UPDATED_ANALYZER_CACHE_KEY: [cache_artifact]})
    self.assertEqual({labels.CACHE_OUTPUT_PATH_LABEL: '/dummy'}, result)

  def testGetStatusOutputPathsEntries(self):
    # disabled.
    self.assertEmpty(executor_utils.GetStatsOutputPathEntries(True, {}))

    # enabled.
    pre_transform_stats = standard_artifacts.ExampleStatistics()
    pre_transform_stats.uri = '/pre_transform_stats'
    pre_transform_schema = standard_artifacts.Schema()
    pre_transform_schema.uri = '/pre_transform_schema'
    post_transform_anomalies = standard_artifacts.ExampleAnomalies()
    post_transform_anomalies.uri = '/post_transform_anomalies'
    post_transform_stats = standard_artifacts.ExampleStatistics()
    post_transform_stats.uri = '/post_transform_stats'
    post_transform_schema = standard_artifacts.Schema()
    post_transform_schema.uri = '/post_transform_schema'

    result = executor_utils.GetStatsOutputPathEntries(
        False, {
            standard_component_specs.PRE_TRANSFORM_STATS_KEY:
                [pre_transform_stats],
            standard_component_specs.PRE_TRANSFORM_SCHEMA_KEY:
                [pre_transform_schema],
            standard_component_specs.POST_TRANSFORM_ANOMALIES_KEY:
                [post_transform_anomalies],
            standard_component_specs.POST_TRANSFORM_STATS_KEY:
                [post_transform_stats],
            standard_component_specs.POST_TRANSFORM_SCHEMA_KEY:
                [post_transform_schema],
        })
    self.assertEqual(
        {
            labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL:
                '/pre_transform_stats',
            labels.PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL:
                '/pre_transform_schema',
            labels.POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL:
                '/post_transform_anomalies',
            labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL:
                '/post_transform_stats',
            labels.POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL:
                '/post_transform_schema',
        }, result)

  def testGetStatusOutputPathsEntriesMissingArtifact(self):
    pre_transform_stats = standard_artifacts.ExampleStatistics()
    pre_transform_stats.uri = '/pre_transform_stats'

    with self.assertRaisesRegex(
        ValueError, 'all stats_output_paths should be specified or none'):
      executor_utils.GetStatsOutputPathEntries(False, {
          standard_component_specs.PRE_TRANSFORM_STATS_KEY:
              [pre_transform_stats]
      })


if __name__ == '__main__':
  tf.test.main()
