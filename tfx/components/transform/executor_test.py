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
"""Tests for tfx.components.transform.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.transform import executor
from tfx.utils import types


# TODO(b/122478841): Add more detailed tests.
class ExecutorTest(tf.test.TestCase):

  def test_do(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    train_artifact = types.TfxType('ExamplesPath', split='train')
    train_artifact.uri = os.path.join(source_data_dir, 'csv_example_gen/train/')
    eval_artifact = types.TfxType('ExamplesPath', split='eval')
    eval_artifact.uri = os.path.join(source_data_dir, 'csv_example_gen/eval/')
    schema_artifact = types.TfxType('Schema')
    schema_artifact.uri = os.path.join(source_data_dir, 'schema_gen/')

    module_file = os.path.join(source_data_dir,
                               'module_file/transform_module.py')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    transformed_output = types.TfxType('TransformPath')
    transformed_output.uri = os.path.join(output_data_dir, 'transformed_output')
    transformed_train_examples = types.TfxType('ExamplesPath', split='train')
    transformed_train_examples.uri = os.path.join(output_data_dir, 'train')
    transformed_eval_examples = types.TfxType('ExamplesPath', split='eval')
    transformed_eval_examples.uri = os.path.join(output_data_dir, 'eval')
    temp_path_output = types.TfxType('TempPath')
    temp_path_output.uri = tempfile.mkdtemp()

    input_dict = {
        'input_data': [train_artifact, eval_artifact],
        'schema': [schema_artifact],
    }
    output_dict = {
        'transform_output': [transformed_output],
        'transformed_examples': [
            transformed_train_examples, transformed_eval_examples
        ],
        'temp_path': [temp_path_output],
    }

    exec_properties = {
        'module_file': module_file,
    }

    transform_executor = executor.Executor()
    transform_executor.Do(input_dict, output_dict, exec_properties)
    self.assertNotEqual(
        0, len(tf.gfile.ListDirectory(transformed_train_examples.uri)))
    self.assertNotEqual(
        0, len(tf.gfile.ListDirectory(transformed_eval_examples.uri)))
    path_to_saved_model = os.path.join(
        transformed_output.uri, tft.TFTransformOutput.TRANSFORM_FN_DIR,
        tf.saved_model.constants.SAVED_MODEL_FILENAME_PB)
    self.assertTrue(tf.gfile.Exists(path_to_saved_model))


if __name__ == '__main__':
  tf.test.main()
