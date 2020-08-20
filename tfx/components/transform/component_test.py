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
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self.examples = channel_utils.as_channel([examples_artifact])
    self.schema = channel_utils.as_channel(
        [standard_artifacts.Schema()])

  def _verify_outputs(self, transform, materialize=True):
    self.assertEqual(standard_artifacts.TransformGraph.TYPE_NAME,
                     transform.outputs['transform_graph'].type_name)
    if materialize:
      self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                       transform.outputs['transformed_examples'].type_name)
    else:
      self.assertNotIn('transformed_examples', transform.outputs.keys())

  def testConstructFromModuleFile(self):
    module_file = '/path/to/preprocessing.py'
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        module_file=module_file,
    )
    self._verify_outputs(transform)
    self.assertEqual(module_file, transform.spec.exec_properties['module_file'])

  def testConstructWithParameter(self):
    module_file = data_types.RuntimeParameter(name='module-file', ptype=Text)
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        module_file=module_file,
    )
    self._verify_outputs(transform)
    self.assertJsonEqual(
        str(module_file), str(transform.spec.exec_properties['module_file']))

  def testConstructFromPreprocessingFn(self):
    preprocessing_fn = 'path.to.my_preprocessing_fn'
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn=preprocessing_fn,
    )
    self._verify_outputs(transform)
    self.assertEqual(preprocessing_fn,
                     transform.spec.exec_properties['preprocessing_fn'])

  def testConstructWithMaterializationDisabled(self):
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn='my_preprocessing_fn',
        materialize=False)
    self._verify_outputs(transform, materialize=False)

  def testConstructFromPreprocessingFnWithCustomConfig(self):
    preprocessing_fn = 'path.to.my_preprocessing_fn'
    custom_config = {'param': 1}
    transform = component.Transform(
        examples=self.examples,
        schema=self.schema,
        preprocessing_fn=preprocessing_fn,
        custom_config=custom_config,
    )
    self._verify_outputs(transform)
    self.assertEqual(preprocessing_fn,
                     transform.spec.exec_properties['preprocessing_fn'])
    self.assertEqual(json.dumps(custom_config),
                     transform.spec.exec_properties['custom_config'])

  def testConstructMissingUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
      )

  def testConstructDuplicateUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
          module_file='/path/to/preprocessing.py',
          preprocessing_fn='path.to.my_preprocessing_fn',
      )

  def testConstructWithMaterializationDisabledButOutputExamples(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          examples=self.examples,
          schema=self.schema,
          preprocessing_fn='my_preprocessing_fn',
          materialize=False,
          transformed_examples=channel_utils.as_channel(
              [standard_artifacts.Examples()]))

if __name__ == '__main__':
  tf.test.main()
