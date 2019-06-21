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
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.utils import channel
from tfx.utils import types


class SlackComponentSpec(base_component.ComponentSpec):
  """ComponentSpec for Custom TFX Slack Component."""

  COMPONENT_NAME = 'SlackComponent'
  PARAMETERS = {
      'slack_token': ExecutionParameter(type=Text),
      'channel_id': ExecutionParameter(type=Text),
      'timeout_sec': ExecutionParameter(type=int),
  }
  INPUTS = {
      'model_export': ChannelParameter(type_name='ModelExportPath'),
      'model_blessing': ChannelParameter(type_name='ModelBlessingPath'),
  }
  OUTPUTS = {
      'slack_blessing': ChannelParameter(type_name='ModelBlessingPath'),
  }


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
  """

  SPEC_CLASS = SlackComponentSpec
  EXECUTOR_CLASS = executor.Executor

  def __init__(self,
               model_export: channel.Channel,
               model_blessing: channel.Channel,
               slack_token: Text,
               channel_id: Text,
               timeout_sec: int,
               slack_blessing: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    """Construct a SlackComponent.

    Args:
      model_export: A Channel of 'ModelExportPath' type, usually produced by
        Trainer component.
      model_blessing: A Channel of 'ModelBlessingPath' type, usually produced by
        ModelValidator component.
      slack_token: A token used for setting up connection with Slack server.
      channel_id: Slack channel id to communicate on.
      timeout_sec: Seconds to wait for response before default to reject.
      slack_blessing: Optional output channel of 'ModelBlessingPath' with result
        of blessing; will be created for you if not specified.
      name: Optional unique name. Necessary if multiple Pusher components are
        declared in the same pipeline.
    """
    slack_blessing = slack_blessing or channel.Channel(
        type_name='ModelBlessingPath',
        static_artifact_collection=[types.TfxArtifact('ModelBlessingPath')])
    spec = SlackComponentSpec(
        slack_token=slack_token,
        channel_id=channel_id,
        timeout_sec=timeout_sec,
        model_export=model_export,
        model_blessing=model_blessing,
        slack_blessing=slack_blessing)
    super(SlackComponent, self).__init__(spec=spec, name=name)
