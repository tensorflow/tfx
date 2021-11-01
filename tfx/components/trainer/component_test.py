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

import tensorflow as tf
from tfx.components.trainer import component
from tfx.components.trainer import executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import trainer_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.examples = channel_utils.as_channel([standard_artifacts.Examples()])
    self.transform_graph = channel_utils.as_channel(
        [standard_artifacts.TransformGraph()])
    self.schema = channel_utils.as_channel([standard_artifacts.Schema()])
    self.hyperparameters = channel_utils.as_channel(
        [standard_artifacts.HyperParameters()])
    self.train_args = trainer_pb2.TrainArgs(splits=['train'], num_steps=100)
    self.eval_args = trainer_pb2.EvalArgs(splits=['eval'], num_steps=50)

  def _verify_outputs(self, trainer):
    self.assertEqual(
        standard_artifacts.Model.TYPE_NAME,
        trainer.outputs[standard_component_specs.MODEL_KEY].type_name)
    self.assertEqual(
        standard_artifacts.ModelRun.TYPE_NAME,
        trainer.outputs[standard_component_specs.MODEL_RUN_KEY].type_name)

  def testConstructFromModuleFile(self):
    module_file = '/path/to/module/file'
    trainer = component.Trainer(
        module_file=module_file,
        examples=self.examples,
        transform_graph=self.transform_graph,
        schema=self.schema,
        custom_config={'test': 10})
    self._verify_outputs(trainer)
    self.assertEqual(
        module_file,
        trainer.spec.exec_properties[standard_component_specs.MODULE_FILE_KEY])
    self.assertEqual(
        '{"test": 10}', trainer.spec.exec_properties[
            standard_component_specs.CUSTOM_CONFIG_KEY])

  def testConstructWithParameter(self):
    module_file = data_types.RuntimeParameter(name='module-file', ptype=str)
    n_steps = data_types.RuntimeParameter(name='n-steps', ptype=int)
    trainer = component.Trainer(
        module_file=module_file,
        examples=self.examples,
        transform_graph=self.transform_graph,
        schema=self.schema,
        train_args=dict(splits=['train'], num_steps=n_steps),
        eval_args=dict(splits=['eval'], num_steps=n_steps))
    self._verify_outputs(trainer)
    self.assertJsonEqual(
        str(module_file),
        str(trainer.spec.exec_properties[
            standard_component_specs.MODULE_FILE_KEY]))

  def testConstructFromTrainerFn(self):
    trainer_fn = 'path.to.my_trainer_fn'
    trainer = component.Trainer(
        trainer_fn=trainer_fn,
        examples=self.examples,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(
        trainer_fn,
        trainer.spec.exec_properties[standard_component_specs.TRAINER_FN_KEY])

  def testConstructFromRunFn(self):
    run_fn = 'path.to.my_run_fn'
    trainer = component.Trainer(
        run_fn=run_fn,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            executor.GenericExecutor),
        examples=self.examples,
        transform_graph=self.transform_graph,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(
        run_fn,
        trainer.spec.exec_properties[standard_component_specs.RUN_FN_KEY])

  def testConstructWithoutTransformOutput(self):
    module_file = '/path/to/module/file'
    trainer = component.Trainer(
        module_file=module_file,
        examples=self.examples,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(
        module_file,
        trainer.spec.exec_properties[standard_component_specs.MODULE_FILE_KEY])

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
          transform_graph=self.transform_graph,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

  def testConstructDuplicateUserModule(self):
    with self.assertRaises(ValueError):
      _ = component.Trainer(
          module_file='/path/to/module/file',
          trainer_fn='path.to.my_trainer_fn',
          examples=self.examples,
          transform_graph=self.transform_graph,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

    with self.assertRaises(ValueError):
      _ = component.Trainer(
          module_file='/path/to/module/file',
          run_fn='path.to.my_run_fn',
          examples=self.examples,
          transform_graph=self.transform_graph,
          schema=self.schema,
          train_args=self.train_args,
          eval_args=self.eval_args)

  def testConstructWithHParams(self):
    trainer = component.Trainer(
        trainer_fn='path.to.my_trainer_fn',
        examples=self.examples,
        transform_graph=self.transform_graph,
        schema=self.schema,
        hyperparameters=self.hyperparameters,
        train_args=self.train_args,
        eval_args=self.eval_args)
    self._verify_outputs(trainer)
    self.assertEqual(
        standard_artifacts.HyperParameters.TYPE_NAME,
        trainer.inputs[standard_component_specs.HYPERPARAMETERS_KEY].type_name)

  def testConstructWithRuntimeParam(self):
    eval_args = data_types.RuntimeParameter(
        name='eval-args',
        default='{"num_steps": 50}',
        ptype=str,
    )
    custom_config = data_types.RuntimeParameter(
        name='custom-config',
        default='{"test": 10}',
        ptype=str,
    )
    trainer = component.Trainer(
        trainer_fn='path.to.my_trainer_fn',
        examples=self.examples,
        train_args=self.train_args,
        eval_args=eval_args,
        custom_config=custom_config)
    self._verify_outputs(trainer)
    self.assertIsInstance(
        trainer.spec.exec_properties[standard_component_specs.EVAL_ARGS_KEY],
        data_types.RuntimeParameter)
    self.assertIsInstance(
        trainer.spec.exec_properties[
            standard_component_specs.CUSTOM_CONFIG_KEY],
        data_types.RuntimeParameter)


if __name__ == '__main__':
  tf.test.main()
