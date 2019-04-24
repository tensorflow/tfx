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
"""Generic TFX trainer executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.base import base_executor
from tfx.orchestration.gcp import cmle_runner
from tfx.proto import trainer_pb2
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import types
from google.protobuf import json_format


def _all_files_pattern(file_pattern):
  return '{}*'.format(file_pattern)


class Executor(base_executor.BaseExecutor):
  """Generic TFX trainer executor."""

  _CHECKPOINT_FILE_NAME = 'checkpoint'

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Runs trainer job the given input.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - transformed_examples: Transformed example.
        - transform_output: Input transform graph.
        - schema: Schema of the data.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: Exported model.
      exec_properties: A dict of execution properties.
        - train_args: JSON string of trainer_pb2.TrainArgs instance, providing
          args for training.
        - eval_args: JSON string of trainer_pb2.EvalArgs instance, providing
          args for eval.
        - module_file: Python module file containing UDF model definition.
        - warm_starting: Whether or not we need to do warm starting.
        - warm_start_from: Optional. If warm_starting is True, this is the
          directory to find previous model to warm start on.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # TODO(khaas): Move this to tfx/extensions.
    if exec_properties.get('custom_config', None):
      cmle_args = exec_properties.get('custom_config',
                                      {}).get('cmle_training_args')
      if cmle_args:
        return cmle_runner.start_cmle_training(input_dict, output_dict,
                                               exec_properties, cmle_args)

    trainer_fn = io_utils.import_func(exec_properties['module_file'],
                                      'trainer_fn')

    # Set up training parameters
    train_files = [
        _all_files_pattern(
            types.get_split_uri(input_dict['transformed_examples'], 'train'))
    ]
    transform_output = types.get_single_uri(input_dict['transform_output'])
    eval_files = _all_files_pattern(
        types.get_split_uri(input_dict['transformed_examples'], 'eval'))
    schema_file = io_utils.get_only_uri_in_dir(
        types.get_single_uri(input_dict['schema']))

    train_args = trainer_pb2.TrainArgs()
    eval_args = trainer_pb2.EvalArgs()
    json_format.Parse(exec_properties['train_args'], train_args)
    json_format.Parse(exec_properties['eval_args'], eval_args)

    # https://github.com/tensorflow/tfx/issues/45: Replace num_steps=0 with
    # num_steps=None.  Conversion of the proto to python will set the default
    # value of an int as 0 so modify the value here.  Tensorflow will raise an
    # error if num_steps <= 0.
    train_steps = train_args.num_steps or None
    eval_steps = eval_args.num_steps or None

    output_path = types.get_single_uri(output_dict['output'])
    serving_model_dir = path_utils.serving_model_dir(output_path)
    eval_model_dir = path_utils.eval_model_dir(output_path)

    # Assemble warm start path if needed.
    warm_start_from = None
    if exec_properties.get('warm_starting') and exec_properties.get(
        'warm_start_from'):
      previous_model_dir = os.path.join(exec_properties['warm_start_from'],
                                        path_utils.SERVING_MODEL_DIR)
      if previous_model_dir and tf.gfile.Exists(
          os.path.join(previous_model_dir, self._CHECKPOINT_FILE_NAME)):
        warm_start_from = previous_model_dir

    # TODO(b/126242806) Use PipelineInputs when it is available in third_party.
    hparams = tf.contrib.training.HParams(
        train_files=train_files,
        transform_output=transform_output,
        output_dir=output_path,
        serving_model_dir=serving_model_dir,
        eval_files=eval_files,
        schema_file=schema_file,
        train_steps=train_steps,
        eval_steps=eval_steps,
        warm_start_from=warm_start_from)

    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())

    training_spec = trainer_fn(hparams, schema)

    # Train the model
    tf.logging.info('Training model.')
    tf.estimator.train_and_evaluate(training_spec['estimator'],
                                    training_spec['train_spec'],
                                    training_spec['eval_spec'])
    tf.logging.info('Training complete.  Model written to %s',
                    serving_model_dir)

    # Export an eval savedmodel for TFMA
    tf.logging.info('Exporting eval_savedmodel for TFMA.')
    tfma.export.export_eval_savedmodel(
        estimator=training_spec['estimator'],
        export_dir_base=eval_model_dir,
        eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

    tf.logging.info('Exported eval_savedmodel to %s.', eval_model_dir)
