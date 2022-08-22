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
"""Tests for tfx.components.example_diff.component."""
import tensorflow as tf
from tfx.components.example_diff import component
from tfx.proto import example_diff_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples1 = standard_artifacts.Examples()
    examples1.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    examples2 = standard_artifacts.Examples()
    examples2.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'foo'])
    include_split_pairs = [('train', 'train'), ('eval', 'foo')]
    config = example_diff_pb2.ExampleDiffConfig(
        paired_example_skew=example_diff_pb2.PairedExampleSkew(
            skew_sample_size=9))
    example_diff = component.ExampleDiff(
        examples_test=channel_utils.as_channel([examples1]),
        examples_base=channel_utils.as_channel([examples2]),
        config=config,
        include_split_pairs=include_split_pairs)
    self.assertEqual(
        standard_artifacts.ExamplesDiff.TYPE_NAME, example_diff.outputs[
            standard_component_specs.EXAMPLE_DIFF_RESULT_KEY].type_name)
    self.assertEqual(
        example_diff.spec.exec_properties[
            standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY],
        '[["train", "train"], ["eval", "foo"]]')
    restored_config = example_diff.exec_properties[
        standard_component_specs.EXAMPLE_DIFF_CONFIG_KEY]
    self.assertEqual(restored_config, config)


if __name__ == '__main__':
  tf.test.main()
