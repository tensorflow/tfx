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
"""Tests for tfx.components.transform.executor with Parquet file format."""
import os
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tfx.components.transform import executor
from tfx.components.transform import executor_test
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.utils import io_utils
from tfx_bsl.coders import example_coder

from tensorflow_metadata.proto.v0 import schema_pb2


def _copy_examples_as_parquet(examples_path: str, output_paths: List[str],
                              schema: schema_pb2.Schema):
  """Reads a file with tf.Examples, converts and writes it in Parquet format."""
  if tf.executing_eagerly():
    examples = [
        e.numpy() for e in tf.data.TFRecordDataset(
            examples_path, compression_type='GZIP')
    ]
  else:
    examples = list(
        tf.compat.v1.io.tf_record_iterator(
            examples_path, tf.io.TFRecordOptions(compression_type='GZIP')))
  record_batch = example_coder.ExamplesToRecordBatchDecoder(
      schema.SerializeToString()).DecodeBatch(examples)
  table = pa.Table.from_batches([record_batch])
  for path in output_paths:
    pq.write_table(table, path)


class ExecutorOnParquetTest(executor_test.ExecutorTest):
  # Should not rely on inherited _SOURCE_DATA_DIR for integration tests to work
  # when TFX is installed as a non-editable package.
  _SOURCE_DATA_DIR = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata')
  _FILE_FORMAT = executor._FILE_FORMAT_PARQUET
  _PAYLOAD_FORMAT = example_gen_pb2.PayloadFormat.FORMAT_PARQUET

  def _get_dataset_size(self, files: List[str]) -> int:
    result = 0
    for file in files:
      table = pq.read_table(file)
      result += table.num_rows
    return result

  @classmethod
  def setUpClass(cls):
    super(executor_test.ExecutorTest, cls).setUpClass()
    source_example_dir = os.path.join(cls._SOURCE_DATA_DIR, 'csv_example_gen')

    example_files = fileio.glob(os.path.join(source_example_dir, '*', '*'))
    schema_path = os.path.join(cls._SOURCE_DATA_DIR, 'schema_gen',
                               'schema.pbtxt')
    schema = io_utils.SchemaReader().read(schema_path)
    # Read the example files, convert to parquet and make two copies.
    for filepath in example_files:
      directory, filename = os.path.split(filepath)
      _, parent_dir_name = os.path.split(directory)
      dest_dir1 = os.path.join(cls._ARTIFACT1_URI, parent_dir_name)
      dest_dir2 = os.path.join(cls._ARTIFACT2_URI, parent_dir_name)
      fileio.makedirs(dest_dir1)
      fileio.makedirs(dest_dir2)
      dest_path1 = os.path.join(dest_dir1, filename[:-3] + '.parquet')
      dest_path2 = os.path.join(dest_dir2, filename[:-3] + '.parquet')
      _copy_examples_as_parquet(filepath, [dest_path1, dest_path2], schema)

    # Duplicate the number of train and eval records such that
    # second artifact has twice as many as first.
    artifact2_pattern = os.path.join(cls._ARTIFACT2_URI, '*', '*')
    artifact2_files = fileio.glob(artifact2_pattern)
    for filepath in artifact2_files:
      directory, filename = os.path.split(filepath)
      io_utils.copy_file(filepath, os.path.join(directory, 'dup_' + filename))


if __name__ == '__main__':
  tf.test.main()
