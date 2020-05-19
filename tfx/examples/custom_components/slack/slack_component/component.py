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
"""Example of a TFX custom component integrating with slack.

This component along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
from typing import Any, Dict, NamedTuple, Text

import absl
from slackclient import SlackClient

from tfx.components.util import model_utils
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types import standard_artifacts
from tfx.utils import io_utils


# Case-insensitive text messages that are accepted as signal for approving a
# model.
_APPROVE_TEXT = ['lgtm', 'approve']
# Case-insensitive text messages that are accepted as signal for rejecting a
# model.
_DECLINE_TEXT = ['decline', 'reject']
# Template for notifying model review
_NOTIFY_MODEL_REVIEW_TEMPLATE = """
Please review the model in the following URI: {}"""
# Template for notifying valid model review reply
_NOTIFY_CORRECT_REPLY_TEMPLATE = """
Unrecognized text: "{{}}", please use one of the following to approve:
{}
or one of the following to reject:
{}""".format(_APPROVE_TEXT, _DECLINE_TEXT)


class Timeout(object):
  """Helper class for handle function timeout."""

  def __init__(self, seconds):
    self.seconds = seconds

  def handle_timeout(self, unused_signum, unused_frame):
    msg = 'Did not get model evaluation result in %d seconds' % self.seconds
    absl.logging.warning(msg)
    raise TimeoutError(msg)  # pylint: disable=undefined-variable

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, unused_type, unused_value, unused_traceback):
    signal.alarm(0)


# NamedTuple for slack response.
_SlackResponse = NamedTuple(
    '_SlackResponse',
    [
        # Whether the model is approved.
        ('approved', bool),
        # The user that made the decision.
        ('user_id', Text),
        # The decision message.
        ('message', Text),
        # The slack channel that the decision is made on.
        ('slack_channel_id', Text),
        # The slack thread that the decision is made on.
        ('thread_ts', Text)
    ])


def _is_valid_message(payload: Dict[Text, Any],
                      expected_slack_channel_id: Text,
                      expected_thread_timestamp: int):
  """Evaluates whether a payload is valid.

  A payload is considered valid iff:
    a. it is from the expected slack channel
    b. it is from the expected slack thread
    c. it contains message info

  Args:
    payload: the payload to be evaluated
    expected_slack_channel_id: the id of the expected slack channel
    expected_thread_timestamp: the timestamp of the expected slack thread

  Returns:

  """
  return (payload.get('type') == 'message' and
          payload.get('channel') == expected_slack_channel_id and
          payload.get('text') and
          payload.get('thread_ts') == expected_thread_timestamp)


def _fetch_slack_blessing(slack_token: Text, slack_channel_id: Text,
                          model_uri: Text) -> _SlackResponse:
  """Send message via Slack channel and wait for response.

  Args:
    slack_token: The user-defined function to obtain token to send and receive
      messages.
    slack_channel_id: The id of the Slack channel to send and receive
      messages.
    model_uri: The URI of the model waiting for human review.

  Returns:
    A _SlackResponse instance.

  Raises:
    ConnectionError:
      When connection to slack server cannot be established.

  """
  sc = SlackClient(slack_token)
  msg = _NOTIFY_MODEL_REVIEW_TEMPLATE.format(model_uri)
  ts = 0
  if not sc.rtm_connect():
    msg = 'Cannot connect to slack server with given token'
    absl.logging.error(msg)
    raise ConnectionError(msg)  # pylint: disable=undefined-variable

  sc.rtm_send_message(slack_channel_id, message=msg)

  while sc.server.connected:
    payload_list = sc.rtm_read()
    if not payload_list:
      continue

    for payload in payload_list:
      if payload.get('ok') and payload.get('reply_to') == 0 and not ts:
        ts = payload['ts']
        continue
      if not _is_valid_message(payload, slack_channel_id, ts):
        continue
      if payload.get('text').lower() in _APPROVE_TEXT:
        absl.logging.info('User %s approves the model located at %s',
                          payload.get('user'), model_uri)
        return _SlackResponse(True, payload.get('user'), payload.get('text'),
                              slack_channel_id, str(ts))
      elif payload.get('text').lower() in _DECLINE_TEXT:
        absl.logging.info('User %s declines the model located at %s',
                          payload.get('user'), model_uri)
        return _SlackResponse(False, payload.get('user'), payload.get('text'),
                              slack_channel_id, str(ts))
      else:
        unrecognized_text = payload.get('text')
        absl.logging.info('Unrecognized response: %s', unrecognized_text)
        sc.rtm_send_message(
            slack_channel_id,
            message=_NOTIFY_CORRECT_REPLY_TEMPLATE.format(unrecognized_text),
            thread=ts)


@component
def SlackComponent(  # pylint: disable=invalid-name
    model: InputArtifact[standard_artifacts.Model],
    model_blessing: InputArtifact[standard_artifacts.ModelBlessing],
    slack_blessing: OutputArtifact[standard_artifacts.ModelBlessing],
    slack_token: Parameter[Text],
    slack_channel_id: Parameter[Text],
    timeout_sec: Parameter[int]):
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

  Args:
    model: Input of type `standard_artifacts.Model`, usually produced by
      a Trainer component.
    model_blessing: Input of type `standard_artifacts.ModelBlessing`,
      usually produced by a ModelValidator component.
    slack_blessing: Output of type `standard_artifacts.ModelBlessing` with
      result of blessing; will be created for you if not specified.
    slack_token: A token used for setting up connection with Slack server.
    slack_channel_id: Slack channel id to communicate on.
    timeout_sec: Seconds to wait for response before default to reject.
  """
  # We only consider a model as blessed if both of the following conditions
  # are met:
  # - The model is blessed by model validator. This is determined by looking
  #   for file named 'BLESSED' from the output from Model Validator.
  # - The model is blessed by a human reviewer. This logic is in
  #   _fetch_slack_blessing().
  slack_response = None
  with Timeout(timeout_sec):
    if model_utils.is_model_blessed(model_blessing):
      slack_response = _fetch_slack_blessing(slack_token,
                                             slack_channel_id,
                                             model.uri)

  # If model is blessed, write an empty file named 'BLESSED' in the assigned
  # output path. Otherwise, write an empty file named 'NOT_BLESSED' instead.
  if slack_response and slack_response.approved:
    io_utils.write_string_file(
        os.path.join(slack_blessing.uri, 'BLESSED'), '')
    slack_blessing.set_int_custom_property('blessed', 1)
  else:
    io_utils.write_string_file(
        os.path.join(slack_blessing.uri, 'NOT_BLESSED'), '')
    slack_blessing.set_int_custom_property('blessed', 0)
  if slack_response:
    slack_blessing.set_string_custom_property('slack_decision_maker',
                                              slack_response.user_id)
    slack_blessing.set_string_custom_property('slack_decision_message',
                                              slack_response.message)
    slack_blessing.set_string_custom_property('slack_decision_channel',
                                              slack_response.slack_channel_id)
    slack_blessing.set_string_custom_property('slack_decision_thread',
                                              slack_response.thread_ts)
  absl.logging.info('Blessing result written to %s.', slack_blessing.uri)
