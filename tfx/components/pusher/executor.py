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
"""Generic TFX pusher executor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.orchestration.gcp import cmle_runner
from tfx.proto import pusher_pb2
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import types
from google.protobuf import json_format


class Executor(base_executor.BaseExecutor):
  """Generic TFX pusher executor."""

  def Do(self, input_dict,
         output_dict,
         exec_properties):
    """Push model to target if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.
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
    model_export = types.get_single_instance(input_dict['model_export'])
    model_export_uri = model_export.uri
    model_blessing_uri = types.get_single_uri(input_dict['model_blessing'])
    model_push = types.get_single_instance(output_dict['model_push'])
    model_push_uri = model_push.uri
    # TODO(jyzhao): should this be in driver or executor.
    if not tf.gfile.Exists(os.path.join(model_blessing_uri, 'BLESSED')):
      model_push.set_int_custom_property('pushed', 0)
      tf.logging.info('Model on %s was not blessed',)
      return
    tf.logging.info('Model pushing.')
    # Copy the model we are pushing into
    model_path = path_utils.serving_model_path(model_export_uri)
    # Note: we do not have a logical model version right now. This
    # model_version is a timestamp mapped to trainer's exporter.
    model_version = os.path.basename(model_path)
    tf.logging.info('Model version is %s', model_version)
    io_utils.copy_dir(model_path, os.path.join(model_push_uri, model_version))
    tf.logging.info('Model written to %s.', model_push_uri)

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
    if tf.gfile.Exists(serving_path):
      tf.logging.info(
          'Destination directory %s already exists, skipping current push.',
          serving_path)
    else:
      # tf.serving won't load partial model, it will retry until fully copied.
      io_utils.copy_dir(model_path, serving_path)
      tf.logging.info('Model written to serving path %s.', serving_path)

    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property('pushed_model', model_export_uri)
    model_push.set_int_custom_property('pushed_model_id', model_export.id)
    tf.logging.info('Model pushed to %s.', serving_path)

    if exec_properties.get('custom_config'):
      cmle_serving_args = exec_properties.get('custom_config',
                                              {}).get('cmle_serving_args')
      if cmle_serving_args is not None:
        return cmle_runner.deploy_model_for_serving(serving_path, model_version,
                                                    cmle_serving_args)
