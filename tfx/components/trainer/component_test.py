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
"""Tests for tfx.components.trainer.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import tensorflow as tf
from tfx.components.trainer import component
from tfx.orchestration import data_types
from tfx.proto import trainer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()

    self.examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self.transform_output = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])
    self.train_args = trainer_pb2.TrainArgs(num_steps=100)
    self.eval_args = trainer_pb2.EvalArgs(num_steps=50)

  def _verify_outputs(self, trainer):
    self.assertEqual(standard_artifacts.Model.TYPE_NAME,
                     trainer.outputs['model'].type_name)

  def testConstructFromModuleFile(self):
    module_file = '/path/to/module/file'
    trainer = component.Trainer(
        module_file=module_file,
        transformed_examples=self.examples,
        transform_graph=self.transform_output,
        schema=self.schema,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(module_file, trainer.spec.exec_properties['module_file'])

  def testConstructWithParameter(self):
    module_file = data_types.RuntimeParameter(name='module-file', ptype=Text)
    n_steps = data_types.RuntimeParameter(name='n-steps', ptype=int)
    trainer = component.Trainer(
        module_file=module_file,
        transformed_examples=self.examples,
        transform_graph=self.transform_output,
        schema=self.schema,
        train_args=dict(num_steps=n_steps),
        eval_args=dict(num_steps=n_steps))
    self._verify_outputs(trainer)
    self.assertJsonEqual(
        str(module_file), str(trainer.spec.exec_properties['module_file']))

  def testConstructFromTrainerFn(self):
    trainer_fn = 'path.to.my_trainer_fn'
    trainer = component.Trainer(
        trainer_fn=trainer_fn,
        transformed_examples=self.examples,
        transform_graph=self.transform_output,
        schema=self.schema,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(trainer_fn, trainer.spec.exec_properties['trainer_fn'])

  def testConstructWithoutTransformOutput(self):
    module_file = '/path/to/module/file'
    trainer = component.Trainer(
        module_file=module_file,
        examples=self.examples,
        schema=self.schema,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(module_file, trainer.spec.exec_properties['module_file'])

  def testConstructDuplicateExamples(self):
    with self.assertRaises(ValueError):
      _ = component.Trainer(
          module_file='/path/to/module/file',
          examples=self.examples,
          transformed_examples=self.examples,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

  def testConstructMissingTransformOutput(self):
    with self.assertRaises(ValueError):
      _ = component.Trainer(
          module_file='/path/to/module/file',
          transformed_examples=self.examples,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

  def testConstructMissingUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Trainer(
          examples=self.examples,
          transform_graph=self.transform_output,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

  def testConstructDuplicateUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Trainer(
          module_file='/path/to/module/file',
          trainer_fn='path.to.my_trainer_fn',
          examples=self.examples,
          transform_graph=self.transform_output,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)


if __name__ == '__main__':
  tf.test.main()
