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
"""A BigShuffle workflow."""

from __future__ import absolute_import

import os
import random
import string
import tempfile
import time

import apache_beam as beam
from tfx.benchmarks import benchmark_base

from google.cloud import storage


class BigShuffleBenchmarkBase(benchmark_base.BenchmarkBase):
  """Contains standalone Beam pipeline benchmarks.

  This benchmark is not dependent on outside datasets or Tensorflow
  dependencies.
  """

  def __init__(self,
               file_size=1e6,
               input_file='input.txt',
               output_file='output.txt'):
    super(BigShuffleBenchmarkBase, self).__init__()
    self._input_file = input_file
    self._output_file = output_file
    self.file_size = file_size  # 1e8 bytes = 100 MB

    self._chars_per_line = 100

    self._read_write_to_cloud = False
    if self._input_file.startswith('gs://'):
      self._read_write_to_cloud = True

  def regenerate_data(self, file_size=None):
    if self.file_size:
      self.file_size = file_size

    if not self._read_write_to_cloud:
      open(self._input_file, 'w').close()

    self._generate_data()

  def _upload_tmp_file_to_bucket(self, tmp_input_file):
    storage_client = storage.Client()

    path_parts = self._input_file[5:].split('/')

    bucket_name = path_parts[0]
    destination_blob_name = path_parts[-1]

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(tmp_input_file)
    os.remove(tmp_input_file)

  def _generate_data(self):
    file_to_open = self._input_file

    if self._read_write_to_cloud:
      file_to_open = tempfile.NamedTemporaryFile(suffix='.txt').name

    with open(file_to_open, 'w') as f:
      letters = string.ascii_lowercase
      num_lines = int(self.file_size / self._chars_per_line)

      for _ in range(0, num_lines):
        line = ''.join(
            random.choice(letters) for i in range(self._chars_per_line)) + '\n'
        f.write(line)

    if self._read_write_to_cloud:
      self._upload_tmp_file_to_bucket(file_to_open)

  def benchmark_big_shuffle(self):
    """Run the BigShuffle benchmark."""
    p = self._create_beam_pipeline()

    # Read the text file
    lines = p | beam.io.textio.ReadFromText(file_pattern=self._input_file)

    # Count the occurrences of each word.
    output = (
        lines
        | beam.Map(str.strip)
        | beam.Map(lambda x: (x[:5], x[5:99]))
        | beam.GroupByKey('group')
        | beam.FlatMap(lambda kv: ['%s%s' % (kv[0], kv[1]) for val in kv[1]]))

    # Write the output
    _ = output | beam.io.textio.WriteToText(self._output_file)

    # Run the pipeline
    start = time.time()
    result = p.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    return delta

  def benchmark_empty_pipeline(self):
    """Creates an empty pipeline to run as a baseline."""
    p = self._create_beam_pipeline()

    # Run the pipeline.
    start = time.time()
    result = p.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    return delta
