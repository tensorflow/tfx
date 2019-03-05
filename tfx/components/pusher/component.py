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
"""TFX Pusher component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.components.pusher import executor
from tfx.proto import pusher_pb2
from tfx.utils import channel
from tfx.utils import types
from google.protobuf import json_format


class Pusher(base_component.BaseComponent):
  """Official TFX Pusher component.

  The `Pusher` component can be used to push an validated SavedModel from output
  of `Trainer` to tensorflow Serving (tf.serving). If the model is not blessed
  by `ModelValidator`, no push will happen.

  Args:
    model_export: A Channel of 'ModelExportPath' type, usually produced by
      Trainer component.
    model_blessing: A Channel of 'ModelBlessingPath' type, usually produced by
      ModelValidator component.
    push_destination: A pusher_py2.PushDestinationLabel instance, providing
      info for tensorflow serving to load models.
    name: Optional unique name. Necessary if multiple Pusher components are
      declared in the same pipeline.
    custom_config: A dict which contains the deployment job parameters to be
      passed to Google Cloud ML Engine.  For the full set of parameters
      supported by Google Cloud ML Engine, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
    outputs: Optional dict from name to output channel.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - model_push: A channel of 'ModelPushPath' with result of push.
  """

  def __init__(self,
               model_export,
               model_blessing,
               push_destination,
               name = None,
               custom_config = None,
               outputs = None):
    component_name = 'Pusher'
    input_dict = {
        'model_export': channel.as_channel(model_export),
        'model_blessing': channel.as_channel(model_blessing),
    }
    exec_properties = {
        'push_destination': json_format.MessageToJson(push_destination),
        'custom_config': custom_config,
    }
    super(Pusher, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self):
    """Creates outputs for Pusher.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    model_push_artifact_collection = [types.TfxType('ModelPushPath',)]
    return base_component.ComponentOutputs({
        'model_push':
            channel.Channel(
                type_name='ModelPushPath',
                static_artifact_collection=model_push_artifact_collection),
    })

  def _type_check(self, input_dict,
                  exec_properties):
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unused right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    input_dict['model_export'].type_check('ModelExportPath')
    input_dict['model_blessing'].type_check('ModelBlessingPath')
