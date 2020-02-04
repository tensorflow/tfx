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
"""TFX pusher executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

import absl
import tensorflow as tf

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.proto import pusher_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Key for model in executor input_dict.
MODEL_KEY = 'model'
# Key for model blessing in executor input_dict.
MODEL_BLESSING_KEY = 'model_blessing'

# Key for pushed model in executor output_dict.
PUSHED_MODEL_KEY = 'pushed_model'


class Executor(base_executor.BaseExecutor):
  """TFX Pusher executor to push the new TF model to a filesystem target.

  The Pusher component is used to deploy a validated model to a filesystem
  target or serving environment using tf.serving.  Pusher depends on the outputs
  of ModelValidator to determine if a model is ready to push. A model is
  considered to be safe to push only if ModelValidator has marked it as BLESSED.
  A push action delivers the model exports produced by Trainer to the
  destination defined in the ``push_destination`` of the component config.

  To include Pusher in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L104.

  For more details on tf.serving itself, please refer to
  https://tensorflow.org/tfx/guide/pusher.  For a tutuorial on TF Serving,
  please refer to https://www.tensorflow.org/tfx/guide/serving.
  """

  def CheckBlessing(self, input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]]) -> bool:
    """Check that model is blessed by upstream ModelValidator, or update output.

    Args:
      input_dict: Input dict from input key to a list of artifacts:
        - model_blessing: model blessing path from model_validator. Pusher looks
          for a file named 'BLESSED' to consider the model blessed and safe to
          push.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one.

    Returns:
      True if the model is blessed by validator.
    """
    model_blessing = artifact_utils.get_single_instance(
        input_dict[MODEL_BLESSING_KEY])
    model_push = artifact_utils.get_single_instance(
        output_dict[PUSHED_MODEL_KEY])
    # TODO(jyzhao): should this be in driver or executor.
    if not model_utils.is_model_blessed(model_blessing):
      model_push.set_int_custom_property('pushed', 0)
      absl.logging.info('Model on %s was not blessed', model_blessing.uri)
      return False
    return True

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Push model to target directory if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.  A push
          action delivers the model exports produced by Trainer to the
          destination defined in component config.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: A dict of execution properties, including:
        - push_destination: JSON string of pusher_pb2.PushDestination instance,
          providing instruction of destination to push model.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    if not self.CheckBlessing(input_dict, output_dict):
      return
    model_push = artifact_utils.get_single_instance(
        output_dict[PUSHED_MODEL_KEY])
    model_push_uri = model_push.uri
    model_export = artifact_utils.get_single_instance(input_dict[MODEL_KEY])
    model_export_uri = model_export.uri
    absl.logging.info('Model pushing.')
    # Copy the model we are pushing into
    model_path = path_utils.serving_model_path(model_export_uri)
    # Note: we do not have a logical model version right now. This
    # model_version is a timestamp mapped to trainer's exporter.
    model_version = os.path.basename(model_path)
    absl.logging.info('Model version is %s', model_version)
    io_utils.copy_dir(model_path, os.path.join(model_push_uri, model_version))
    absl.logging.info('Model written to %s.', model_push_uri)

    # Copied to a fixed outside path, which can be listened by model server.
    #
    # If model is already successfully copied to outside before, stop copying.
    # This is because model validator might blessed same model twice (check
    # mv driver) with different blessing output, we still want Pusher to
    # handle the mv output again to keep metadata tracking, but no need to
    # copy to outside path again..
    # TODO(jyzhao): support rpc push and verification.
    push_destination = pusher_pb2.PushDestination()
    json_format.Parse(exec_properties['push_destination'], push_destination)
    serving_path = os.path.join(push_destination.filesystem.base_directory,
                                model_version)
    if tf.io.gfile.exists(serving_path):
      absl.logging.info(
          'Destination directory %s already exists, skipping current push.',
          serving_path)
    else:
      # tf.serving won't load partial model, it will retry until fully copied.
      io_utils.copy_dir(model_path, serving_path)
      absl.logging.info('Model written to serving path %s.', serving_path)

    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property('pushed_model', model_export_uri)
    model_push.set_int_custom_property('pushed_model_id', model_export.id)
    absl.logging.info('Model pushed to %s.', serving_path)
