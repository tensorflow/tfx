# Copyright 2026 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.experimental.filter.executor."""

import os
from typing import List
import tensorflow as tf
from tfx.components.experimental.filter import executor

from tfx.types import standard_artifacts
from tfx.types import artifact_utils


def dummy_filter_fn(serialized_example: bytes) -> bool:
  """A simple filter function that parses the example and filters by age."""
  example = tf.train.Example()
  example.ParseFromString(serialized_example)
  features = example.features.feature
  if 'age' in features:
    return features['age'].int64_list.value[0] > 18
  return False


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._input_data_dir = os.path.join(self.get_temp_dir(), 'input')

  def _create_test_examples(self, examples_artifact: standard_artifacts.Examples, split_name: str, values: List[int]):
    """Creates a TFRecord file with test examples for the given split."""
    split_dir = artifact_utils.get_split_uri([examples_artifact], split_name)
    tf.io.gfile.makedirs(split_dir)
    file_path = os.path.join(split_dir, 'data.tfrecord.gz')

    options = tf.io.TFRecordOptions(compression_type='GZIP')
    with tf.io.TFRecordWriter(file_path, options=options) as writer:
      for val in values:
        example = tf.train.Example()
        example.features.feature['age'].int64_list.value.append(val)
        writer.write(example.SerializeToString())

  def testExecutor(self):
    # 1. Prepare input and output examples artifacts.
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.uri = self._input_data_dir
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    # 2. Create input data:
    # train split has ages [10, 20, 30] -> filtered should have [20, 30]
    # eval split has ages [15, 25] -> filtered should have [25]
    self._create_test_examples(examples_artifact, 'train', [10, 20, 30])
    self._create_test_examples(examples_artifact, 'eval', [15, 25])

    filtered_examples_artifact = standard_artifacts.Examples()
    filtered_examples_artifact.uri = self._output_data_dir

    input_dict = {'examples': [examples_artifact]}
    output_dict = {'filtered_examples': [filtered_examples_artifact]}

    # Full python import path to the dummy_filter_fn
    filter_fn_path = (
        'tfx.components.experimental.filter.executor_test.dummy_filter_fn')

    exec_properties = {'filter_fn_path': filter_fn_path}

    # 3. Run the executor.
    filter_executor = executor.Executor()
    filter_executor.Do(input_dict, output_dict, exec_properties)

    # 4. Verify output splits.
    decoded_splits = artifact_utils.decode_split_names(
        filtered_examples_artifact.split_names)
    self.assertEqual(decoded_splits, ['train', 'eval'])

    # 5. Verify the content of the filtered train split.
    train_output_dir = artifact_utils.get_split_uri(
        [filtered_examples_artifact], 'train')
    train_output_files = tf.io.gfile.glob(os.path.join(train_output_dir, '*'))
    self.assertNotEmpty(train_output_files)

    train_ages = []
    # Read the output TFRecords. Beam writes sharded files, so we read all matching files.
    for file_path in train_output_files:
      raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
      for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        train_ages.append(example.features.feature['age'].int64_list.value[0])

    self.assertCountEqual(train_ages, [20, 30])

    # 6. Verify the content of the filtered eval split.
    eval_output_dir = artifact_utils.get_split_uri(
        [filtered_examples_artifact], 'eval')
    eval_output_files = tf.io.gfile.glob(os.path.join(eval_output_dir, '*'))
    self.assertNotEmpty(eval_output_files)

    eval_ages = []
    for file_path in eval_output_files:
      raw_dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
      for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        eval_ages.append(example.features.feature['age'].int64_list.value[0])

    self.assertCountEqual(eval_ages, [25])


if __name__ == '__main__':
  tf.test.main()
