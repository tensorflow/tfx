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
"""Custom executor to push TFX model to AI Platform."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.types import artifact_utils
from tfx.utils import path_utils


# Keys to the items in custom_config passed as a part of exec_properties.
SERVING_ARGS_KEY = 'ai_platform_serving_args'


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to Google Cloud AI Platform serving."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
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
      ValueError:
        If ai_platform_serving_args is not in exec_properties.custom_config.
        If Serving model path does not start with gs://.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    model_push = artifact_utils.get_single_instance(
        output_dict[tfx_pusher_executor.PUSHED_MODEL_KEY])
    if not self.CheckBlessing(input_dict):
      model_push.set_int_custom_property('pushed', 0)
      return

    model_export = artifact_utils.get_single_instance(
        input_dict[tfx_pusher_executor.MODEL_KEY])
    model_export_uri = model_export.uri

    exec_properties_copy = exec_properties.copy()
    custom_config = exec_properties_copy.pop('custom_config', {})
    ai_platform_serving_args = custom_config[SERVING_ARGS_KEY]
    if not ai_platform_serving_args:
      raise ValueError(
          '\'ai_platform_serving_args\' is missing in \'custom_config\'')
    # Deploy the model.
    model_path = path_utils.serving_model_path(model_export_uri)
    # Note: we do not have a logical model version right now. This
    # model_version is a timestamp mapped to trainer's exporter.
    model_version = os.path.basename(model_path)
    executor_class_path = '%s.%s' % (self.__class__.__module__,
                                     self.__class__.__name__)
    runner.deploy_model_for_aip_prediction(
        model_path,
        model_version,
        ai_platform_serving_args,
        executor_class_path,
    )

    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property('pushed_model', model_path)
