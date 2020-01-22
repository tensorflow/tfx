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
"""Tests for tfx.examples.custom_components.tuner.tuner_component.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.examples.custom_components.tuner.tuner_component import component
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class TunerTest(tf.test.TestCase):

  def setUp(self):
    super(TunerTest, self).setUp()

    self.examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])

  def _verify_output(self, tuner):
    self.assertEqual(standard_artifacts.Model.TYPE_NAME,
                     tuner.outputs['model'].type_name)
    self.assertEqual(standard_artifacts.HyperParameters.TYPE_NAME,
                     tuner.outputs['best_hyperparameters'].type_name)

  def testConstructWithModuleFile(self):
    tuner = component.Tuner(
        examples=self.examples,
        schema=self.schema,
        module_file='/path/to/module/file')
    self._verify_output(tuner)

  def testConstructWithTunerFn(self):
    tuner = component.Tuner(
        examples=self.examples, schema=self.schema, tuner_fn='path.to.tuner_fn')
    self._verify_output(tuner)

  def testConstructDuplicateUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Tuner(
          examples=self.examples,
          schema=self.schema,
          module_file='/path/to/module/file',
          tuner_fn='path.to.tuner_fn')


if __name__ == '__main__':
  tf.test.main()
