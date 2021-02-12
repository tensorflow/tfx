# Lint as: python2, python3
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
"""Tests for tfx.components.schema_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tfx.components.schema_gen import executor
from tfx.dsl.io import fileio
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class ExecutorTest(tf.test.TestCase):

  def testDo(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.uri = os.path.join(source_data_dir, 'statistics_gen')
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    schema_output = standard_artifacts.Schema()
    schema_output.uri = os.path.join(output_data_dir, 'schema_output')

    input_dict = {
        standard_component_specs.STATISTICS_KEY: [statistics_artifact],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.EXCLUDE_SPLITS_KEY:
            json_utils.dumps(['test'])
    }

    output_dict = {
        standard_component_specs.SCHEMA_KEY: [schema_output],
    }

    schema_gen_executor = executor.Executor()
    schema_gen_executor.Do(input_dict, output_dict, exec_properties)
    self.assertNotEqual(0, len(fileio.listdir(schema_output.uri)))


if __name__ == '__main__':
  tf.test.main()
