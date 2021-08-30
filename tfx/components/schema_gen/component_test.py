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

import tensorflow as tf
from tfx.components.schema_gen import component
from tfx.orchestration import data_types
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class SchemaGenTest(tf.test.TestCase):

  def testConstruct(self):
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    exclude_splits = ['eval']
    schema_gen = component.SchemaGen(
        statistics=channel_utils.as_channel([statistics_artifact]),
        exclude_splits=exclude_splits)
    self.assertEqual(
        standard_artifacts.Schema.TYPE_NAME,
        schema_gen.outputs[standard_component_specs.SCHEMA_KEY].type_name)
    self.assertTrue(schema_gen.spec.exec_properties[
        standard_component_specs.INFER_FEATURE_SHAPE_KEY])
    self.assertEqual(
        schema_gen.spec.exec_properties[
            standard_component_specs.EXCLUDE_SPLITS_KEY], '["eval"]')

  def testConstructWithParameter(self):
    statistics_artifact = standard_artifacts.ExampleStatistics()
    statistics_artifact.split_names = artifact_utils.encode_split_names(
        ['train'])
    infer_shape = data_types.RuntimeParameter(name='infer-shape', ptype=int)
    schema_gen = component.SchemaGen(
        statistics=channel_utils.as_channel([statistics_artifact]),
        infer_feature_shape=infer_shape)
    self.assertEqual(
        standard_artifacts.Schema.TYPE_NAME,
        schema_gen.outputs[standard_component_specs.SCHEMA_KEY].type_name)
    self.assertJsonEqual(
        str(schema_gen.spec.exec_properties[
            standard_component_specs.INFER_FEATURE_SHAPE_KEY]),
        str(infer_shape))


if __name__ == '__main__':
  tf.test.main()
