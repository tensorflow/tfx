# Copyright 2022 Google LLC. All Rights Reserved.
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
"""module file for object detection training util functions."""

import os
import pprint

from absl import logging
import tensorflow as tf
import tfx.v1 as tfx

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.core import task_factory
from official.core import train_lib
from official.modeling import hyperparams
from official.vision.beta.projects.yolo.common import registry_imports  # pylint: disable=unused-import
from official.vision.beta.projects.yolo.configs import yolo  # pylint: disable=unused-import
from official.vision.beta.projects.yolo.serving import export_module_factory
from official.vision.serving import export_saved_model_lib


_INPUT_TYPE = 'tf_example'
_BATCH_SIZE = 1
_INPUT_IMAGE_SIZE = [224, 224]


@exp_factory.register_config_factory('tfx_scaled_yolo')
def tfx_scaled_yolo() -> cfg.ExperimentConfig:
  """Create ExperimentConfig, and override inputs from trainer.

  Returns:
    ExperimentConfig initialized from YAML sources.
  """
  experiment_config = yolo.scaled_yolo()
  config_file_path = os.path.join(
      os.path.dirname(__file__), 'scaled_yolo_config.yaml')
  experiment_config = hyperparams.override_params_dict(
      experiment_config, config_file_path, is_strict=True)
  return experiment_config


def overwrite_experiment_config(
    fn_args: tfx.components.FnArgs) -> cfg.ExperimentConfig:
  """Overwrite the model garden task config.

  Args:
    fn_args: TFX trainer inputs, including inputs-by-split, hparams from
      service config, export directory, etc.

  Returns:
    Updated ExperimentConfig after updates.
  """
  experiment_config = tfx_scaled_yolo()
  experiment_config.task.train_data.input_path = fn_args.train_files
  experiment_config.task.validation_data.input_path = fn_args.eval_files
  experiment_config.trainer.train_steps = fn_args.train_steps

  return experiment_config


def run_fn(fn_args: tfx.components.FnArgs):
  """TFX trainer entry point."""

  params = overwrite_experiment_config(fn_args)

  params.validate()
  params.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s',
               pp.pformat(params.as_dict()))

  export_dir = fn_args.serving_model_dir

  # strategy = tf.distribute.OneDeviceStrategy(
  #     'device:CPU:0')
  strategy = tf.distribute.MirroredStrategy()
  model_dir = fn_args.model_run_dir
  with strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  mode = 'train_and_eval'
  model, _ = train_lib.run_experiment(
      distribution_strategy=strategy,
      task=task,
      mode=mode,
      params=params,
      model_dir=model_dir)

  model.summary(print_fn=logging.info)
  logging.info('Training Complete!')

  export_dir = fn_args.serving_model_dir
  # Sets checkpoint path
  checkpoint_dir = model_dir
  if params.trainer.best_checkpoint_export_subdir:
    checkpoint_dir = os.path.join(
        model_dir, params.trainer.best_checkpoint_export_subdir)

  export_module = export_module_factory.get_export_module(
      params=params,
      input_type=_INPUT_TYPE,
      batch_size=_BATCH_SIZE,
      input_image_size=_INPUT_IMAGE_SIZE,
      num_channels=3)

  def postprocessing_fn(prediction):
    concated_input = tf.concat([
        prediction['detection_boxes'],
        tf.expand_dims(prediction['detection_classes'], 2),
        tf.expand_dims(prediction['detection_scores'], 2)
    ], -1)
    return {'post_processing_pred': concated_input}

  postprocessing_func = postprocessing_fn
  export_module.postprocessor = postprocessing_func

  export_saved_model_lib.export_inference_graph(
      input_type=_INPUT_TYPE,
      batch_size=_BATCH_SIZE,
      input_image_size=_INPUT_IMAGE_SIZE,
      params=params,
      checkpoint_path=checkpoint_dir,
      export_dir=export_dir,
      export_module=export_module)
