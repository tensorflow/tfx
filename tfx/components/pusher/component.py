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

from typing import Any, Dict, Optional, Text, Type

from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.components.base.base_component import Parameter
from tfx.components.pusher import executor
from tfx.proto import pusher_pb2
from tfx.utils import channel
from tfx.utils import types


class PusherSpec(base_component.ComponentSpec):
  """Pusher component spec."""

  COMPONENT_NAME = 'Pusher'
  PARAMETERS = [
      Parameter('push_destination', type=pusher_pb2.PushDestination,
                optional=True),
      Parameter('custom_config', type=Dict[Text, Any], optional=True),
  ]
  INPUTS = [
      ChannelInput('model_export', type='ModelExportPath'),
      ChannelInput('model_blessing', type='ModelBlessingPath'),
  ]
  OUTPUTS = [
      ChannelOutput('model_push', type='ModelPushPath'),
  ]


# TODO(b/133845381): Investigate other ways to keep push destination converged.
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
    push_destination: A pusher_pb2.PushDestination instance, providing
      info for tensorflow serving to load models. Optional if executor_class
      doesn't require push_destination.
    name: Optional unique name. Necessary if multiple Pusher components are
      declared in the same pipeline.
    custom_config: A dict which contains the deployment job parameters to be
      passed to Google Cloud ML Engine.  For the full set of parameters
      supported by Google Cloud ML Engine, refer to
      https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
    executor_class: Optional custom python executor class.
    model_push: Optional output 'ModelPushPath' channel with result of push.
  """

  def __init__(self,
               model_export: channel.Channel,
               model_blessing: channel.Channel,
               push_destination: Optional[pusher_pb2.PushDestination] = None,
               name: Text = None,
               custom_config: Optional[Dict[Text, Any]] = None,
               executor_class: Optional[Type[
                   base_executor.BaseExecutor]] = executor.Executor,
               model_push: Optional[channel.Channel] = None):
    if not model_push:
      model_push = channel.Channel(
          type_name='ModelPushPath',
          static_artifact_collection=[types.TfxArtifact('ModelPushPath')])
    if push_destination is None:
      if executor_class == executor.Executor:
        raise ValueError('push_destination is required unless custom '
                         'executor_class is supplied that does not require it.')
    spec = PusherSpec(
        model_export=channel.as_channel(model_export),
        model_blessing=channel.as_channel(model_blessing),
        push_destination=push_destination,
        custom_config=custom_config,
        model_push=model_push)

    super(Pusher, self).__init__(
        unique_name=name,
        spec=spec,
        executor=executor_class)
