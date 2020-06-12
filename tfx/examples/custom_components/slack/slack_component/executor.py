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
"""Example of a TFX custom executor integrating with slack.

This executor along with other custom component related code will only serve as
an example and will not be supported by TFX team.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
from typing import Any, Dict, List, NamedTuple, Text

import absl
from slackclient import SlackClient

from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.types import artifact_utils
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


class Executor(base_executor.BaseExecutor):
  """Executor for Slack component."""

  def _is_valid_message(self, payload: Dict[Text, Any],
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

  def _fetch_slack_blessing(self, slack_token: Text, slack_channel_id: Text,
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
        if not self._is_valid_message(payload, slack_channel_id, ts):
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

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Get human review result on a model through Slack channel.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from evaluator.
      output_dict: Output dict from key to a list of artifacts, including:
        - slack_blessing: model blessing result.
      exec_properties: A dict of execution properties, including:
        - slack_token: Token used to setup connection with slack server.
        - slack_channel_id: The id of the Slack channel to send and receive
          messages.
        - timeout_sec: How long do we wait for response, in seconds.

    Returns:
      None

    Raises:
      TimeoutError:
        When there is no decision made within timeout_sec.
      ConnectionError:
        When connection to slack server cannot be established.

    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Fetch execution properties from exec_properties dict.
    slack_token = exec_properties['slack_token']
    slack_channel_id = exec_properties['slack_channel_id']
    timeout_sec = exec_properties['timeout_sec']

    # Fetch input URIs from input_dict.
    model_export_uri = artifact_utils.get_single_uri(input_dict['model'])
    model_blessing = artifact_utils.get_single_instance(
        input_dict['model_blessing'])

    # Fetch output artifact from output_dict.
    slack_blessing = artifact_utils.get_single_instance(
        output_dict['slack_blessing'])

    # We only consider a model as blessed if both of the following conditions
    # are met:
    # - The model is blessed by evaluator. This is determined by looking
    #   for file named 'BLESSED' from the output from Evaluator.
    # - The model is blessed by a human reviewer. This logic is in
    #   _fetch_slack_blessing().
    slack_response = None
    with Timeout(timeout_sec):
      if model_utils.is_model_blessed(model_blessing):
        slack_response = self._fetch_slack_blessing(slack_token,
                                                    slack_channel_id,
                                                    model_export_uri)

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
