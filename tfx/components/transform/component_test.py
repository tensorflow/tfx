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
"""Tests for tfx.components.transform.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from typing import Text

import tensorflow as tf
from tfx.components.transform import component
from tfx.orchestration import data_types
from tfx.proto import transform_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self.examples = channel_utils.as_channel([examples_artifact])
    self.schema = channel_utils.as_channel(
        [standard_artifacts.Schema()])

  def _verify_outputs(self,
                      transform,
                      materialize=True,
                      disable_analyzer_cache=False):
    self.assertEqual(
        standard_artifacts.TransformGraph.TYPE_NAME, transform.outputs[
            standard_component_specs.TRANSFORM_GRAPH_KEY].type_name)
    if materialize:
      self.assertEqual(
          standard_artifacts.Examples.TYPE_NAME, transform.outputs[
              standard_component_specs.TRANSFORMED_EXAMPLES_KEY].type_name)
    else:
      self.assertNotIn(standard_component_specs.TRANSFORMED_EXAMPLES_KEY,
                       transform.outputs.keys())

    if disable_analyzer_cache:
      self.assertNotIn('updated_analyzer_cache', transform.outputs.keys())
    else:
      self.assertEqual(
          standard_artifacts.TransformCache.TYPE_NAME, transform.outputs[
              standard_component_specs.UPDATED_ANALYZER_CACHE_KEY].type_name)

  def test_construct_from_module_file(self):
    module_file = '/path/to/preprocessing.py'
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        module_file=module_file,
    )
    self._verify_outputs(transform)
    self.assertEqual(
        module_file,
        transform.exec_properties[standard_component_specs.MODULE_FILE_KEY])

  def test_construct_with_parameter(self):
    module_file = data_types.RuntimeParameter(name='module-file', ptype=Text)
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        module_file=module_file,
    )
    self._verify_outputs(transform)
    self.assertJsonEqual(
        str(module_file),
        str(transform.exec_properties[
            standard_component_specs.MODULE_FILE_KEY]))

  def test_construct_from_preprocessing_fn(self):
    preprocessing_fn = 'path.to.my_preprocessing_fn'
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn=preprocessing_fn,
    )
    self._verify_outputs(transform)
    self.assertEqual(
        preprocessing_fn, transform.exec_properties[
            standard_component_specs.PREPROCESSING_FN_KEY])

  def test_construct_with_materialization_disabled(self):
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn='my_preprocessing_fn',
        materialize=False)
    self._verify_outputs(transform, materialize=False)

  def test_construct_with_cache_disabled(self):
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn='my_preprocessing_fn',
        disable_analyzer_cache=True)
    self._verify_outputs(transform, disable_analyzer_cache=True)

  def test_construct_from_preprocessing_fn_with_custom_config(self):
    preprocessing_fn = 'path.to.my_preprocessing_fn'
    custom_config = {'param': 1}
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn=preprocessing_fn,
        custom_config=custom_config,
    )
    self._verify_outputs(transform)
    self.assertEqual(
        preprocessing_fn, transform.spec.exec_properties[
            standard_component_specs.PREPROCESSING_FN_KEY])
    self.assertEqual(
        json.dumps(custom_config), transform.spec.exec_properties[
            standard_component_specs.CUSTOM_CONFIG_KEY])

  def test_construct_missing_user_module(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
      )

  def test_construct_duplicate_user_module(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
          module_file='/path/to/preprocessing.py',
          preprocessing_fn='path.to.my_preprocessing_fn',
      )

  def test_construct_with_splits_config(self):
    splits_config = transform_pb2.SplitsConfig(
        analyze=['train'], transform=['eval'])
    module_file = '/path/to/preprocessing.py'
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        module_file=module_file,
        splits_config=splits_config,
    )
    self._verify_outputs(transform)
    self.assertEqual(
        proto_utils.proto_to_json(splits_config),
        transform.exec_properties[standard_component_specs.SPLITS_CONFIG_KEY])

  def test_construct_with_cache_disabled_but_input_cache(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
          preprocessing_fn='my_preprocessing_fn',
          disable_analyzer_cache=True,
          analyzer_cache=channel_utils.as_channel(
              [standard_artifacts.TransformCache()]))

  def test_construct_with_force_tf_compat_v1_override(self):
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn='my_preprocessing_fn',
        force_tf_compat_v1=True,
    )
    self._verify_outputs(transform)
    self.assertEqual(
        True,
        bool(transform.spec.exec_properties[
            standard_component_specs.FORCE_TF_COMPAT_V1_KEY]))


if __name__ == '__main__':
  tf.test.main()
