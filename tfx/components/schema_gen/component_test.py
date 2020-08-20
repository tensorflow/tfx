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
"""Tests for tfx.components.schema_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.schema_gen import component
from tfx.orchestration import data_types
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class SchemaGenTest(tf.test.TestCase):

  def testConstruct(self):
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    exclude_splits = ['eval']
    schema_gen = component.SchemaGen(
        statistics=channel_utils.as_channel([statistics_artifact]),
        exclude_splits=exclude_splits)
    self.assertEqual(standard_artifacts.Schema.TYPE_NAME,
                     schema_gen.outputs['schema'].type_name)
    self.assertFalse(schema_gen.spec.exec_properties['infer_feature_shape'])
    self.assertEqual(schema_gen.spec.exec_properties['exclude_splits'],
                     '["eval"]')

  def testConstructWithParameter(self):
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train'])
    infer_shape = data_types.RuntimeParameter(name='infer-shape', ptype=bool)
    schema_gen = component.SchemaGen(
        statistics=channel_utils.as_channel([statistics_artifact]),
        infer_feature_shape=infer_shape)
    self.assertEqual(standard_artifacts.Schema.TYPE_NAME,
                     schema_gen.outputs['schema'].type_name)
    self.assertJsonEqual(
        str(schema_gen.spec.exec_properties['infer_feature_shape']),
        str(infer_shape))


if __name__ == '__main__':
  tf.test.main()
