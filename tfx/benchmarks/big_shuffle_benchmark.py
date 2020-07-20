"""A BigShuffle workflow."""

from __future__ import absolute_import

import time
import random
import string
import os
import apache_beam as beam
from google.cloud import storage

from tfx.benchmarks import benchmark_base

class BigShuffleBenchmarkBase(benchmark_base.BenchmarkBase):
  """Contains standalone Beam pipeline benchmarks, not dependent on outside
     datasets and tensorflow dependencies."""

  def __init__(self, file_size=1e6, input_file="input.txt",
               output_file="output.txt"):
    super(BeamPipelineBenchmarkChicagoTaxi, self).__init__()
    self.input_file = input_file
    self.output_file = output_file
    self.file_size = file_size # 1e8 bytes = 100 MB

    self.chars_per_line = 100
    self.bytes_per_line = (self.chars_per_line * 2)

    self.tmp_input_file = "tmp_input.txt"
    self.read_write_to_cloud = False
    if self.input_file[0:5] == "gs://":
      self.read_write_to_cloud = True

  def regenerate_data(self, file_size=None):
    if self.file_size:
      self.file_size = file_size

    if self.read_write_to_cloud:
      open(self.tmp_input_file, 'w').close()
    else:
      open(self.input_file, 'w').close()

    self._generate_data()

  def _upload_tmp_file_to_bucket(self):
    storage_client = storage.Client()

    path_parts = self.input_file[5:].split('/')

    bucket_name = path_parts[0]
    destination_blob_name = path_parts[-1]

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(self.tmp_input_file)
    os.remove(self.tmp_input_file)

  def _generate_data(self):

    if self.read_write_to_cloud:
      f = open(self.tmp_input_file, "a")
    else:
      f = open(self.input_file, "a")

    letters = string.ascii_lowercase
    num_lines = int(self.file_size / self.bytes_per_line)

    for _ in range(0, num_lines):
      line = ''.join(random.choice(letters) for i in range(self.chars_per_line)) + '\n'
      f.write(line)

    f.close()

    if self.read_write_to_cloud:
      self._upload_tmp_file_to_bucket()

  def benchmarkBigShuffle(self):
    """Creates a large file and sorts it by splitting lines into key value pairs"""
    p = self._create_beam_pipeline()

    # Read the text file
    lines = p | beam.io.textio.ReadFromText(file_pattern=self.input_file)

    # Count the occurrences of each word.
    output = (lines
              | beam.Map(str.strip)
              | beam.Map(lambda x: (x[:5], x[5:99]))
              | beam.GroupByKey('group')
              | beam.FlatMap(lambda kv: ['%s%s' % (kv[0], kv[1]) for val in kv[1]]))

    # Write the output
    _ = output | beam.io.textio.WriteToText(self.output_file)

    # Run the pipeline
    start = time.time()
    result = p.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    return delta

  def benchmarkEmptyPipeline(self):
    """Creates an empty pipeline with no data or transformations to run as a baseline"""
    p = self._create_beam_pipeline()

    # Run the pipeline.
    start = time.time()
    result = p.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    return delta
