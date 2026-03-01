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

from typing import Optional

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.examples.custom_components.slack.slack_component import executor
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class SlackComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Slack Component."""

  PARAMETERS = {
      'slack_token': ExecutionParameter(type=str),
      'slack_channel_id': ExecutionParameter(type=str),
      'timeout_sec': ExecutionParameter(type=int),
  }
  INPUTS = {
      'model': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  OUTPUTS = {
      'slack_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
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

  If the model is approved, an artifact will be created in ML metadata. It will
  be materialized as a file named 'BLESSED' in the directory specified by the
  URI of 'slack_blessing' artifact.
  If the model is rejected, an artifact will be created in ML metadata. It will
  be materialized as a file named 'NOT_BLESSED' in the directory specified by
  the URI of 'slack_blessing' channel.
  If no message indicating approve or reject was is received within given within
  timeout_sec, component will error out. This ensures that model will not be
  pushed and the validation is still retry-able.

  The output artifact might contain the following custom properties:
    - blessed: integer value indicating whether the model is blessed
    - slack_decision_maker: the user id that made the decision.
    - slack_decision_message: the message of the decision
    - slack_decision_channel: the slack channel the decision is made on
    - slack_decision_thread: the slack thread the decision is made on
  """

  SPEC_CLASS = SlackComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               model: types.Channel,
               model_blessing: types.Channel,
               slack_token: str,
               slack_channel_id: str,
               timeout_sec: int,
               slack_blessing: Optional[types.Channel] = None):
    """Construct a SlackComponent.

    Args:
      model: A Channel of type `standard_artifacts.Model`, usually produced by
        a Trainer component.
      model_blessing: A Channel of type `standard_artifacts.ModelBlessing`,
        usually produced by a ModelValidator component.
      slack_token: A token used for setting up connection with Slack server.
      slack_channel_id: Slack channel id to communicate on.
      timeout_sec: Seconds to wait for response before default to reject.
      slack_blessing: Optional output channel of type
        `standard_artifacts.ModelBlessing` with result of blessing; will be
        created for you if not specified.
    """
    slack_blessing = slack_blessing or types.Channel(
        type=standard_artifacts.ModelBlessing)
    spec = SlackComponentSpec(
        slack_token=slack_token,
        slack_channel_id=slack_channel_id,
        timeout_sec=timeout_sec,
        model=model,
        model_blessing=model_blessing,
        slack_blessing=slack_blessing)
    super().__init__(spec=spec)
