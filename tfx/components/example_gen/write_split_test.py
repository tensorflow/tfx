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
"""Tests for tfx.components.example_gen.write_split."""
import os

import apache_beam as beam
from apache_beam.metrics.metric import MetricsFilter
from apache_beam.runners.direct import direct_runner
import tensorflow as tf
from tfx.components.example_gen import write_split
from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2


class WriteSplitTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

  def testWriteSplitCounter_WithFormatUnspecified(self):
    count = 10

    def Pipeline(root):
      data = [tf.train.Example()] * count
      _ = (
          root
          | beam.Create(data)
          | write_split.WriteSplit(self._output_data_dir,
                                   example_gen_pb2.FILE_FORMAT_UNSPECIFIED))

    run_result = direct_runner.DirectRunner().run(Pipeline)
    run_result.wait_until_finish()

    num_instances = run_result.metrics().query(
        MetricsFilter().with_name('num_instances'))

    self.assertTrue(
        fileio.exists(
            os.path.join(self._output_data_dir,
                         'data_tfrecord-00000-of-00001.gz')))
    self.assertTrue(num_instances['counters'])
    self.assertEqual(len(num_instances['counters']), 1)
    self.assertEqual(num_instances['counters'][0].result, count)

  def testWriteSplitCounter_WithTFRECORDS_GZIP(self):
    count = 10

    def Pipeline(root):
      data = [tf.train.Example()] * count
      _ = (
          root
          | beam.Create(data)
          | write_split.WriteSplit(self._output_data_dir,
                                   example_gen_pb2.FORMAT_TFRECORDS_GZIP))

    run_result = direct_runner.DirectRunner().run(Pipeline)
    run_result.wait_until_finish()

    num_instances = run_result.metrics().query(
        MetricsFilter().with_name('num_instances'))

    self.assertTrue(
        fileio.exists(
            os.path.join(self._output_data_dir,
                         'data_tfrecord-00000-of-00001.gz')))
    self.assertTrue(num_instances['counters'])
    self.assertEqual(len(num_instances['counters']), 1)
    self.assertEqual(num_instances['counters'][0].result, count)


if __name__ == '__main__':
  tf.test.main()
