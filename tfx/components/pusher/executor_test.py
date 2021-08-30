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
"""Tests for tfx.components.pusher.executor."""

import json
import os
import tensorflow as tf

from tfx.components.pusher import executor
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import path_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        standard_component_specs.MODEL_KEY: [self._model_export],
        standard_component_specs.MODEL_BLESSING_KEY: [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    fileio.makedirs(self._model_push.uri)
    self._output_dict = {
        standard_component_specs.PUSHED_MODEL_KEY: [self._model_push],
    }
    self._serving_model_dir = os.path.join(self._output_data_dir,
                                           'serving_model_dir')
    fileio.makedirs(self._serving_model_dir)
    self._exec_properties = self._MakeExecProperties()
    self._executor = executor.Executor()

  def _MakeExecProperties(self, versioning='AUTO'):
    return {
        standard_component_specs.PUSH_DESTINATION_KEY: json.dumps({
            'filesystem': {
                'base_directory': self._serving_model_dir,
                'versioning': versioning
            }
        })
    }

  def assertDirectoryEmpty(self, path):
    self.assertEqual(len(fileio.listdir(path)), 0)

  def assertDirectoryNotEmpty(self, path):
    self.assertGreater(len(fileio.listdir(path)), 0)

  def assertPushed(self):
    self.assertDirectoryNotEmpty(self._serving_model_dir)
    self.assertDirectoryNotEmpty(self._model_push.uri)
    self.assertEqual(1, self._model_push.get_int_custom_property('pushed'))

  def assertNotPushed(self):
    self.assertDirectoryEmpty(self._serving_model_dir)
    self.assertDirectoryEmpty(self._model_push.uri)
    self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

  def testDoBlessed(self):
    # Prepare blessed ModelBlessing.
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)

    # Run executor with blessed.
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)

    # Check model successfully pushed.
    self.assertPushed()
    version = self._model_push.get_string_custom_property('pushed_version')
    self.assertTrue(version.isdigit())
    self.assertEqual(
        self._model_push.get_string_custom_property('pushed_destination'),
        os.path.join(self._serving_model_dir, version))

  def testDoNotBlessed(self):
    # Prepare not blessed ModelBlessing.
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/not_blessed')
    self._model_blessing.set_int_custom_property('blessed', 0)

    # Run executor with not blessed.
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)

    # Check model not pushed.
    self.assertNotPushed()

  def testDo_ModelBlessedAndInfraBlessed_Pushed(self):
    # Prepare blessed ModelBlessing and blessed InfraBlessing.
    self._model_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    input_dict = {standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing]}
    input_dict.update(self._input_dict)

    # Run executor
    self._executor.Do(input_dict, self._output_dict, self._exec_properties)

    # Check model is pushed.
    self.assertPushed()

  def testDo_InfraNotBlessed_NotPushed(self):
    # Prepare blessed ModelBlessing and **not** blessed InfraBlessing.
    self._model_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.set_int_custom_property('blessed', 0)  # Not blessed.
    input_dict = {standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing]}
    input_dict.update(self._input_dict)

    # Run executor
    self._executor.Do(input_dict, self._output_dict, self._exec_properties)

    # Check model is not pushed.
    self.assertNotPushed()

  def testDo_NoModelBlessing_InfraBlessed_Pushed(self):
    # Prepare successful InfraBlessing only (without ModelBlessing).
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.set_int_custom_property('blessed', 1)  # Blessed.
    input_dict = {
        standard_component_specs.MODEL_KEY:
            self._input_dict[standard_component_specs.MODEL_KEY],
        standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing],
    }

    # Run executor
    self._executor.Do(input_dict, self._output_dict, self._exec_properties)

    # Check model is pushed.
    self.assertPushed()

  def testDo_NoModelBlessing_InfraNotBlessed_NotPushed(self):
    # Prepare unsuccessful InfraBlessing only (without ModelBlessing).
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.set_int_custom_property('blessed', 0)  # Not blessed.
    input_dict = {
        standard_component_specs.MODEL_KEY:
            self._input_dict[standard_component_specs.MODEL_KEY],
        standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing],
    }

    # Run executor
    self._executor.Do(input_dict, self._output_dict, self._exec_properties)

    # Check model is not pushed.
    self.assertNotPushed()

  def testDo_KerasModelPath(self):
    # Prepare blessed ModelBlessing.
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/keras')
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)

    # Run executor
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)

    # Check model is pushed.
    self.assertPushed()

  def testDo_NoBlessing(self):
    # Input without any blessing.
    input_dict = {standard_component_specs.MODEL_KEY: [self._model_export]}

    # Run executor
    self._executor.Do(input_dict, self._output_dict, self._exec_properties)

    # Check model is pushed.
    self.assertPushed()

  def testDo_NoModel(self):
    with self.assertRaisesRegex(RuntimeError, 'Pusher has no model input.'):
      self._executor.Do(
          {},  # No model and infra_blessing input.
          self._output_dict,
          self._exec_properties)

  def testDo_InfraBlessingAsModel(self):
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.uri = os.path.join(self._output_data_dir, 'infra_blessing')
    infra_blessing.set_int_custom_property('blessed', True)
    infra_blessing.set_int_custom_property('has_model', 1)
    # Create dummy model
    blessed_model_path = path_utils.stamped_model_path(infra_blessing.uri)
    fileio.makedirs(blessed_model_path)
    io_utils.write_string_file(
        os.path.join(blessed_model_path, 'my-model'), '')

    self._executor.Do(
        {standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing]},
        self._output_dict,
        self._exec_properties)

    self.assertPushed()
    self.assertTrue(
        fileio.exists(
            os.path.join(self._model_push.uri, 'my-model')))

  def testDo_InfraBlessingAsModel_FailIfNoWarmup(self):
    infra_blessing = standard_artifacts.InfraBlessing()
    infra_blessing.set_int_custom_property('blessed', True)
    infra_blessing.set_int_custom_property('has_model', 0)

    with self.assertRaisesRegex(
        RuntimeError, 'InfraBlessing does not contain a model'):
      self._executor.Do(
          {standard_component_specs.INFRA_BLESSING_KEY: [infra_blessing]},
          self._output_dict,
          self._exec_properties)

if __name__ == '__main__':
  tf.test.main()
