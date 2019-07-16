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
"""TFX local trainer executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from typing import Any, Dict, List, Text

from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.components.base import base_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.proto import trainer_pb2
from tfx.utils import import_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import types
from google.protobuf import json_format


def _all_files_pattern(file_pattern: Text) -> Text:
  return '{}*'.format(file_pattern)


class Executor(base_executor.BaseExecutor):
  """Local trainer used by the TFX Trainer component.

  The Trainer executor supplements TensorFlow training with a component to
  enable warm-start training of any user-specified tf.estimator. The Trainer is
  a library built on top of TensorFlow that is expected to be integrated into a
  custom user-specified binary.

  To include Trainer in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L104.

  For more details on the Trainer component itself, please refer to
  https://tensorflow.org/tfx/guide/trainer.  For a tutorial on TF Estimator,
  please refer to https://www.tensorflow.org/extend/estimators.

  How to create a trainer callback function to be used by this Trainer executor:
  An estimator can be executed by TFX by first creating a trainer_fn callback
  method that returns an estimator and some additional parameters, similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py#L285.
  This becomes the basis of the new Executor for Trainer. This Executor will
  then train and evaluate this estimator using the
  tf.estimator.train_and_evaluate API to train locally.
  """

  # Name of subdirectory which contains checkpoints from prior runs
  _CHECKPOINT_FILE_NAME = 'checkpoint'

  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Uses a user-supplied tf.estimator to train a TensorFlow model locally.

    The Trainer Executor invokes a training_fn callback function provided by
    the user via the module_file parameter.  With the tf.estimator returned by
    this function, the Trainer Executor then builds a TensorFlow model using the
    user-provided tf.estimator.

    Args:
      input_dict: Input dict from input key to a list of ML-Metadata Artifacts.
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

    Raises:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # TODO(zhitaoli): Deprecate this in a future version.
    if exec_properties.get('custom_config', None):
      cmle_args = exec_properties.get('custom_config',
                                      {}).get('cmle_training_args')
      if cmle_args:
        executor_class_path = '.'.join([Executor.__module__, Executor.__name__])
        tf.logging.warn(
            'Passing \'cmle_training_args\' to trainer directly is deprecated, '
            'please use extension executor at '
            'tfx.extensions.google_cloud_ai_platform.trainer.executor instead')

        return runner.start_cmle_training(input_dict, output_dict,
                                          exec_properties, executor_class_path,
                                          cmle_args)

    trainer_fn = import_utils.import_func_from_source(
        exec_properties['module_file'], 'trainer_fn')

    # Set up training parameters
    train_files = [
        _all_files_pattern(
            types.get_split_uri(input_dict['transformed_examples'], 'train'))
    ]
    transform_output = types.get_single_uri(input_dict['transform_output'])
    eval_files = [
        _all_files_pattern(
            types.get_split_uri(input_dict['transformed_examples'], 'eval'))
    ]
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
        # A list of uris for train files.
        train_files=train_files,
        # A single uri for transform graph produced by TFT.
        transform_output=transform_output,
        # A single uri for the output directory of the serving model.
        serving_model_dir=serving_model_dir,
        # A list of uris for eval files.
        eval_files=eval_files,
        # A single uri for schema file.
        schema_file=schema_file,
        # Number of train steps.
        train_steps=train_steps,
        # Number of eval steps.
        eval_steps=eval_steps,
        # A single uri for the model directory to warm start from.
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
