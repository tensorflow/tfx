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

import json
import os
from typing import Any, Dict, List, Text

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import constants
from tfx.components.trainer import fn_args_utils
from tfx.components.util import udf_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import path_utils

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


def _all_files_pattern(file_pattern: Text) -> Text:
  return os.path.join(file_pattern, '*')


def _is_chief():
  """Returns true if this is run in the master (chief) of training cluster."""
  tf_config = json.loads(os.environ.get(constants.TF_CONFIG_ENV) or '{}')

  # If non distributed mode, current process should always behave as chief.
  if not tf_config or not tf_config.get('cluster', {}):
    return True

  task_type = tf_config['task']['type']
  task_index = tf_config['task']['index']

  # 'master' is a legacy notation of chief node in distributed training flock.
  return task_type == 'chief' or (task_type == 'master' and task_index == 0)


class TrainerFnArgs(dict):
  """Wrapper class to help migrate from contrib.HParam to new data structure."""

  def __getattr__(self, key):
    if key in self:
      return self[key]
    else:
      raise AttributeError('No such attribute: ' + key)

  def __setattr__(self, key, value):
    self[key] = value


class GenericExecutor(base_executor.BaseExecutor):
  """Local generic trainer executor for the TFX Trainer component.

  The Trainer executor supplements TensorFlow training with a component to
  enable warm-start training of any user-specified TF model. The Trainer is
  a library built on top of TensorFlow that is expected to be integrated into a
  custom user-specified binary.

  To include Trainer in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L104.

  For more details on the Trainer component itself, please refer to
  https://tensorflow.org/tfx/guide/trainer.  For a tutorial on Tensorflow,
  please refer to https://www.tensorflow.org/tutorials.

  How to create a trainer callback function to be used by this Trainer executor:
  A model training can be executed by TFX by first creating a run_fn callback
  method that defines, trains an TF Model and saves it to the provided location,
  This becomes the basis of the Executor for GenericTrainer. This Executor will
  then execute the run_fn with correct parameters by resolving the input
  artifacts, output artifacts and execution properties.
  """

  # Name of subdirectory which contains checkpoints from prior runs
  _CHECKPOINT_FILE_NAME = 'checkpoint'

  def _GetFnArgs(self, input_dict: Dict[Text, List[types.Artifact]],
                 output_dict: Dict[Text, List[types.Artifact]],
                 exec_properties: Dict[Text, Any]) -> TrainerFnArgs:
    fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties)

    # Load and deserialize custom config from execution properties.
    # Note that in the component interface the default serialization of custom
    # config is 'null' instead of '{}'. Therefore we need to default the
    # json_utils.loads to 'null' then populate it with an empty dict when
    # needed.
    custom_config = json_utils.loads(
        exec_properties.get(constants.CUSTOM_CONFIG_KEY, 'null')) or {}
    if not isinstance(custom_config, dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict. Got %s instead.' % type(custom_config))

    # TODO(ruoyu): Make this a dict of tag -> uri instead of list.
    if input_dict.get(constants.BASE_MODEL_KEY):
      base_model = path_utils.serving_model_path(
          artifact_utils.get_single_uri(input_dict[constants.BASE_MODEL_KEY]))
    else:
      base_model = None

    if input_dict.get(constants.HYPERPARAMETERS_KEY):
      hyperparameters_file = io_utils.get_only_uri_in_dir(
          artifact_utils.get_single_uri(
              input_dict[constants.HYPERPARAMETERS_KEY]))
      hyperparameters_config = json.loads(
          file_io.read_file_to_string(hyperparameters_file))
    else:
      hyperparameters_config = None

    output_path = artifact_utils.get_single_uri(
        output_dict[constants.MODEL_KEY])
    serving_model_dir = path_utils.serving_model_dir(output_path)
    eval_model_dir = path_utils.eval_model_dir(output_path)

    model_run_dir = artifact_utils.get_single_uri(
        output_dict[constants.MODEL_RUN_KEY])

    # TODO(b/126242806) Use PipelineInputs when it is available in third_party.
    return TrainerFnArgs(
        # A list of uris for train files.
        train_files=fn_args.train_files,
        # An optional single uri for transform graph produced by TFT. Will be
        # None if not specified.
        transform_output=fn_args.transform_graph_path,
        # A single uri for the output directory of the serving model.
        serving_model_dir=serving_model_dir,
        # A single uri for the output directory of the eval model.
        # Note that this is estimator only, Keras doesn't require it for TFMA.
        eval_model_dir=eval_model_dir,
        # A list of uris for eval files.
        eval_files=fn_args.eval_files,
        # A single uri for the output directory of model training related files.
        model_run_dir=model_run_dir,
        # A single uri for schema file.
        schema_file=fn_args.schema_path,
        # Number of train steps.
        train_steps=fn_args.train_steps,
        # Number of eval steps.
        eval_steps=fn_args.eval_steps,
        # Base model that will be used for this training job.
        base_model=base_model,
        # An optional kerastuner.HyperParameters config.
        hyperparameters=hyperparameters_config,
        # A fn_args_utils.DataAccessor. Contains factories that can create
        # tf.data.Datasets or other means to access the train/eval data.
        data_accessor=fn_args.data_accessor,
        # Additional parameters to pass to trainer function.
        **custom_config)

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Uses a user-supplied run_fn to train a TensorFlow model locally.

    The Trainer Executor invokes a run_fn callback function provided by
    the user via the module_file parameter. In this function, user defines the
    model and trains it, then saves the model and training related files
    (e.g, Tensorboard logs) to the provided locations.

    Args:
      input_dict: Input dict from input key to a list of ML-Metadata Artifacts.
        - examples: Examples used for training, must include 'train' and 'eval'
          if custom splits is not specified in train_args and eval_args.
        - transform_output: Optional input transform graph.
        - schema: Schema of the data.
      output_dict: Output dict from output key to a list of Artifacts.
        - model: Exported model.
        - model_run: Model training related outputs (e.g., Tensorboard logs)
      exec_properties: A dict of execution properties.
        - train_args: JSON string of trainer_pb2.TrainArgs instance, providing
          args for training.
        - eval_args: JSON string of trainer_pb2.EvalArgs instance, providing
          args for eval.
        - module_file: Python module file containing UDF model definition.
        - warm_starting: Whether or not we need to do warm starting.
        - warm_start_from: Optional. If warm_starting is True, this is the
          directory to find previous model to warm start on.
        - custom_config: Optional. JSON-serialized dict of additional parameters
          to pass to trainer function.

    Returns:
      None

    Raises:
      ValueError: When neither or both of 'module_file' and 'run_fn'
        are present in 'exec_properties'.
      RuntimeError: If run_fn failed to generate model in desired location.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    fn_args = self._GetFnArgs(input_dict, output_dict, exec_properties)
    run_fn = udf_utils.get_fn(exec_properties, 'run_fn')

    # Train the model
    absl.logging.info('Training model.')
    run_fn(fn_args)

    # Note: If trained with multi-node distribution workers, it is the user
    # module's responsibility to export the model only once.
    if not tf.io.gfile.exists(fn_args.serving_model_dir):
      raise RuntimeError('run_fn failed to generate model.')

    absl.logging.info(
        'Training complete. Model written to %s. ModelRun written to %s',
        fn_args.serving_model_dir, fn_args.model_run_dir)


class Executor(GenericExecutor):
  """Local estimator based trainer executor used by the TFX Trainer component.

  How to create a trainer callback function to be used by this Trainer executor:
  An estimator can be executed by TFX by first creating a trainer_fn callback
  method that returns an estimator and some additional parameters, similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py#L285.
  This becomes the basis of the new Executor for Trainer. This Executor will
  then train and evaluate this estimator using the
  tf.estimator.train_and_evaluate API to train locally.
  """

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
          if custom splits is not specified in train_args and eval_args.
        - transform_output: Optional input transform graph.
        - schema: Schema of the data.
      output_dict: Output dict from output key to a list of Artifacts.
        - model: Exported model.
        - model_run: Model training related outputs (e.g., Tensorboard logs)
      exec_properties: A dict of execution properties.
        - train_args: JSON string of trainer_pb2.TrainArgs instance, providing
          args for training.
        - eval_args: JSON string of trainer_pb2.EvalArgs instance, providing
          args for eval.
        - module_file: Python module file containing UDF model definition.
        - warm_starting: Whether or not we need to do warm starting.
        - warm_start_from: Optional. If warm_starting is True, this is the
          directory to find previous model to warm start on.
        - custom_config: Optional. JSON-serialized dict of additional parameters
          to pass to trainer function.

    Returns:
      None

    Raises:
      ValueError: When neither or both of 'module_file' and 'trainer_fn'
        are present in 'exec_properties'.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    fn_args = self._GetFnArgs(input_dict, output_dict, exec_properties)
    trainer_fn = udf_utils.get_fn(exec_properties, 'trainer_fn')

    schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

    # TODO(b/160795287): Deprecate estimator based executor.
    # Provide user with a modified fn_args, with model_run given as
    # the working directory. Executor will then copy user models to
    # model artifact directory.
    serving_dest = fn_args.serving_model_dir
    eval_dest = fn_args.eval_model_dir

    working_dir = fn_args.model_run_dir
    fn_args.serving_model_dir = path_utils.serving_model_dir(working_dir)
    fn_args.eval_model_dir = path_utils.eval_model_dir(working_dir)

    training_spec = trainer_fn(fn_args, schema)

    # Train the model
    absl.logging.info('Training model.')
    tf.estimator.train_and_evaluate(training_spec['estimator'],
                                    training_spec['train_spec'],
                                    training_spec['eval_spec'])

    absl.logging.info(
        'Training complete. Model written to %s. ModelRun written to %s',
        fn_args.serving_model_dir, fn_args.model_run_dir)

    # Export an eval savedmodel for TFMA. If distributed training, it must only
    # be written by the chief worker, as would be done for serving savedmodel.
    if _is_chief():
      absl.logging.info('Exporting eval_savedmodel for TFMA.')
      tfma.export.export_eval_savedmodel(
          estimator=training_spec['estimator'],
          export_dir_base=fn_args.eval_model_dir,
          eval_input_receiver_fn=training_spec['eval_input_receiver_fn'])

      absl.logging.info('Exported eval_savedmodel to %s.',
                        fn_args.eval_model_dir)

      # TODO(b/160795287): Deprecate estimator based executor.
      # Copy serving and eval model from model_run to model artifact directory.
      serving_source = path_utils.serving_model_path(fn_args.model_run_dir)
      io_utils.copy_dir(serving_source, serving_dest)
      absl.logging.info('Serving model copied to: %s.', serving_dest)

      eval_source = path_utils.eval_model_path(fn_args.model_run_dir)
      io_utils.copy_dir(eval_source, eval_dest)
      absl.logging.info('Eval model copied to: %s.', eval_dest)

    else:
      absl.logging.info(
          'Model export is skipped because this is not the chief worker.')
