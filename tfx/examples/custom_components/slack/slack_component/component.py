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
"""Example of a TFX custom component integrating with slack.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from slack_component import executor
from typing import Optional, Text

from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.utils import channel
from tfx.utils import types


class SlackComponent(base_component.BaseComponent):
  """Custom TFX Slack Component.

  This custom component serves as a bridge between TFX pipeline and human model
  reviewers to enable review-and-push workflow in model development cycle. It
  utilizes Slack API to send message to user-defined Slack channel with model
  URI info and wait for go / no-go decision from the same Slack channel:
    * To approve the model, a user need to reply the thread sent out by the bot
      started by SlackComponent with 'lgtm' or 'approve'.
    * To reject the model, a user need to reply the thread sent out by the bot
      started by SlackComponent with 'decline' or 'reject'.

  Args:
    model_export: A Channel of 'ModelExportPath' type, usually produced by
      Trainer component.
    model_blessing: A Channel of 'ModelBlessingPath' type, usually produced by
      ModelValidator component.
    slack_token: A token used for setting up connection with Slack server.
    channel_id: Slack channel id to communicate on.
    timeout_sec: Seconds to wait for response before default to reject.
    name: Optional unique name. Necessary if multiple Pusher components are
      declared in the same pipeline.
  Attributes:
    outputs: A ComponentOutputs including following keys:
      - blessing: A channel of 'ModelBlessingPath' with result of blessing.
  """

  def __init__(self,
               model_export: channel.Channel,
               model_blessing: channel.Channel,
               slack_token: Text,
               channel_id: Text,
               timeout_sec: int,
               name: Optional[Text] = None,
               outputs: Optional[base_component.ComponentOutputs] = None):
    component_name = 'SlackComponent'
    input_dict = {
        'model_export': channel.as_channel(model_export),
        'model_blessing': channel.as_channel(model_blessing),
    }
    exec_properties = {
        'slack_token': slack_token,
        'channel_id': channel_id,
        'timeout_sec': timeout_sec,
    }
    super(SlackComponent, self).__init__(
        component_name=component_name,
        unique_name=name,
        driver=base_driver.BaseDriver,
        executor=executor.Executor,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties=exec_properties)

  def _create_outputs(self) -> base_component.ComponentOutputs:
    """Creates outputs for SlackComponent.

    Returns:
      ComponentOutputs object containing the dict of [Text -> Channel]
    """
    slack_blessing_output = [types.TfxType('ModelBlessingPath')]
    return base_component.ComponentOutputs({
        'slack_blessing':
            channel.Channel(
                type_name='ModelBlessingPath',
                static_artifact_collection=slack_blessing_output),
    })

  def _type_check(self, input_dict, exec_properties):
    """Does type checking for the inputs and exec_properties.

    Args:
      input_dict: A Dict[Text, Channel] as the inputs of the Component.
      exec_properties: A Dict[Text, Any] as the execution properties of the
        component. Unchecked right now.

    Raises:
      TypeError: if the type_name of given Channel is different from expected.
    """
    del exec_properties
    input_dict['model_export'].type_check('ModelExportPath')
    input_dict['model_blessing'].type_check('ModelBlessingPath')
