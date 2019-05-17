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
"""Custom executor to push TFX model to CMLE serving."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf
from typing import Any, Dict, List, Text
from google.protobuf import json_format
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.orchestration.gcp import cmle_runner
from tfx.proto import pusher_pb2
from tfx.utils import path_utils
from tfx.utils import types


_POLLING_INTERVAL_IN_SECONDS = 30


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to Google Cloud AI Platform serving."""

  def _make_local_temp_destination(self) -> Text:
    """Make a temp destination to push the model."""
    temp_dir = tempfile.mkdtemp()
    push_destination = pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=temp_dir))
    return json_format.MessageToJson(push_destination)

  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]):
    """Overrides the tfx_pusher_executor.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Pusher.executor.  custom_config.ai_platform_serving_args
        is consumed by this class.  For the full set of parameters supported by
        Google Cloud AI Platform, refer to
        https://cloud.google.com/ml-engine/docs/tensorflow/deploying-models#creating_a_model_version.

    Returns:
      None
    Raises:
      ValueError: if ai_platform_serving_args is not in
      exec_properties.custom_config.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    if not self.CheckBlessing(input_dict, output_dict):
      return

    model_export = types.get_single_instance(input_dict['model_export'])
    model_export_uri = model_export.uri
    model_blessing_uri = types.get_single_uri(input_dict['model_blessing'])
    model_push = types.get_single_instance(output_dict['model_push'])
    # TODO(jyzhao): should this be in driver or executor.
    if not tf.gfile.Exists(os.path.join(model_blessing_uri, 'BLESSED')):
      model_push.set_int_custom_property('pushed', 0)
      tf.logging.info('Model on %s was not blessed',)
      return

    exec_properties_copy = exec_properties.copy()
    custom_config = exec_properties_copy.pop('custom_config', {})
    ai_platform_serving_args = custom_config['ai_platform_serving_args']

    # Deploy the model.
    model_path = path_utils.serving_model_path(model_export_uri)
    # Note: we do not have a logical model version right now. This
    # model_version is a timestamp mapped to trainer's exporter.
    model_version = os.path.basename(model_path)
    if ai_platform_serving_args is not None:
      cmle_runner.deploy_model_for_cmle_serving(model_path, model_version,
                                                ai_platform_serving_args)

    # Make sure artifacts are populated in a standard way by calling
    # tfx.pusher.executor.Executor.Do().
    exec_properties_copy['push_destination'] = exec_properties.get(
        'push_destination', self._make_local_temp_destination())
    super(Executor, self).Do(input_dict, output_dict, exec_properties_copy)
