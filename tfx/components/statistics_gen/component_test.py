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
"""Tests for tfx.components.statistics_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_metadata.proto.v0 import problem_statement_pb2
from tfx.components.statistics_gen import component
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    statistics_gen = component.StatisticsGen(
        examples=channel_utils.as_channel([examples]))
    self.assertEqual(standard_artifacts.ExampleStatistics.TYPE_NAME,
                     statistics_gen.outputs['statistics'].type_name)

  def testConstructWithProblemStatementAndSchema(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])
    problem_statement = problem_statement_pb2.ProblemStatement()
    schema = standard_artifacts.Schema()
    statistics_gen = component.StatisticsGen(
        examples=channel_utils.as_channel([examples]),
        problem_statement=problem_statement,
        schema=channel_utils.as_channel([schema]))
    self.assertEqual(standard_artifacts.ExampleStatistics.TYPE_NAME,
                     statistics_gen.outputs['statistics'].type_name)


if __name__ == '__main__':
  tf.test.main()
