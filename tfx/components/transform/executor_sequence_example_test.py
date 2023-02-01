# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.transform.executor with sequnce examples."""

import os
import unittest
import tensorflow as tf
from tfx.components.testdata.module_file import transform_sequence_module
from tfx.components.transform import executor_test
from tfx.proto import example_gen_pb2


@unittest.skipIf(tf.__version__ < '2',
                 'Native SequenceExample support not available with TF1')
class ExecutorWithSequenceExampleTest(executor_test.ExecutorTest):

  # Should not rely on inherited _SOURCE_DATA_DIR for integration tests to work
  # when TFX is installed as a non-editable package.
  _SOURCE_DATA_DIR = os.path.join(
      os.path.dirname(os.path.dirname(__file__)), 'testdata')

  _SOURCE_EXAMPLE_DIR = 'tfrecord_sequence'
  _PAYLOAD_FORMAT = example_gen_pb2.FORMAT_TF_SEQUENCE_EXAMPLE
  _PREPROCESSING_FN = transform_sequence_module.preprocessing_fn
  _STATS_OPTIONS_UPDATER_FN = transform_sequence_module.stats_options_updater_fn
  _SCHEMA_ARTIFACT_DIR = 'schema_gen_sequence'
  _MODULE_FILE = 'module_file/transform_sequence_module.py'

  _TEST_COUNTERS = {
      'num_instances': 25500,
      'total_columns_count': 3,
      'analyze_columns_count': 2,
      'transform_columns_count': 2,
      'metric_committed_sum': 20
  }

  _CACHE_TEST_METRICS = {
      'num_instances_tfx.Transform_1st_run': 25500,
      'num_instances_tfx.Transform_2nd_run': 15000,
      'num_instances_tfx.DataValidation_1st_run': 25500,
      'num_instances_tfx.DataValidation_2nd_run': 25500
  }


if __name__ == '__main__':
  tf.test.main()
