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
"""TFX local trainer executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma

from google.protobuf import json_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.utils import import_utils
from tfx.utils import io_utils
from tfx.utils import path_utils


def _all_files_pattern(file_pattern: Text) -> Text:
  return '{}*'.format(file_pattern)


class TrainerFnArgs(object):
  """Wrapper class to help migrate from contrib.HParam to new data structure."""

  def __init__(self, **kwargs):
    self._data = kwargs

  def __getitem__(self, key):
    return self._data[key]

  def __getattr__(self, key):
    return self._data[key]


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

  def _GetTrainerFn(self, exec_properties: Dict[Text, Any]) -> Any:
    """Loads and returns user-defined trainer_fn."""

    has_module_file = bool(exec_properties.get('module_file'))
    has_trainer_fn = bool(exec_properties.get('trainer_fn'))

    if has_module_file == has_trainer_fn:
      raise ValueError(
          "Neither or both of 'module_file' 'trainer_fn' have been supplied in "
          "'exec_properties'.")

    if has_module_file:
      return import_utils.import_func_from_source(
          exec_properties['module_file'], 'trainer_fn')

    trainer_fn_path_split = exec_properties['trainer_fn'].split('.')
    return import_utils.import_func_from_module(
        '.'.join(trainer_fn_path_split[0:-1]), trainer_fn_path_split[-1])

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Uses a user-supplied tf.estimator to train a TensorFlow model locally.

    The Trainer Executor invokes a training_fn callback function provided by
    the user via the module_file parameter.  With the tf.estimator returned by
    this function, the Trainer Executor then builds a TensorFlow model using the
    user-provided tf.estimator.

    Args:
      input_dict: Input dict from input key to a list of ML-Metadata Artifacts.
        - examples: Examples used for training, must include 'train' and 'eval'
          splits.
        - transform_output: Optional input transform graph.
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
      ValueError: When neither or both of 'module_file' and 'trainer_fn'
        are present in 'exec_properties'.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = exec_properties.get('custom_config') or {}
    if not isinstance(custom_config, dict):
      raise ValueError('Expect custom_config to be a dict but got %s instead' %
                       type(custom_config))

    trainer_fn = self._GetTrainerFn(exec_properties)

    # Set up training parameters
    train_files = [
        _all_files_pattern(
            artifact_utils.get_split_uri(input_dict['examples'], 'train'))
    ]
    transform_output = artifact_utils.get_single_uri(
        input_dict['transform_output']) if input_dict.get(
            'transform_output', None) else None
    eval_files = [
        _all_files_pattern(
            artifact_utils.get_split_uri(input_dict['examples'], 'eval'))
    ]
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict['schema']))
    # TODO(ruoyu): Make this a dict of tag -> uri instead of list.
    base_model = path_utils.serving_model_path(
        artifact_utils.get_single_uri(
            input_dict['base_model'])) if input_dict.get('base_model') else None

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

    output_path = artifact_utils.get_single_uri(output_dict['output'])
    serving_model_dir = path_utils.serving_model_dir(output_path)
    eval_model_dir = path_utils.eval_model_dir(output_path)

    # TODO(b/126242806) Use PipelineInputs when it is available in third_party.
    train_fn_args = TrainerFnArgs(
        # A list of uris for train files.
        train_files=train_files,
        # An optional single uri for transform graph produced by TFT. Will be
        # None if not specified.
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
        # Base model that will be used for this training job.
        base_model=base_model,
        # Additional parameters to pass to trainer function.
        **custom_config)

    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())

    training_spec = trainer_fn(train_fn_args, schema)

    # Train the model
    absl.logging.info('Training model.')
    tf.estimator.train_and_evaluate(training_spec['estimator'],
                                    training_spec['train_spec'],
                                    training_spec['eval_spec'])
    absl.logging.info('Training complete.  Model written to %s',
                      serving_model_dir)

    # Export an eval savedmodel for TFMA
    absl.logging.info('Exporting eval_savedmodel for TFMA.')
    tfma.export.export_eval_savedmodel(
        estimator=training_spec['estimator'],
        export_dir_base=eval_model_dir,
        eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

    absl.logging.info('Exported eval_savedmodel to %s.', eval_model_dir)
