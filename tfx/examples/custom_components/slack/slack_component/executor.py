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

from slackclient import SlackClient

import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.utils import io_utils
from tfx.utils import types

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
    raise TimeoutError()  # pylint: disable=undefined-variable

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)

  def __exit__(self, unused_type, unused_value, unused_traceback):
    signal.alarm(0)


class Executor(base_executor.BaseExecutor):
  """Executor for Slack component."""

  def _fetch_slack_blessing(self, slack_token: Text, channel_id: Text,
                            model_uri: Text) -> bool:
    """Send message via Slack channel and wait for response.

    Args:
      slack_token: The user-defined function to obtain token to send and receive
        messages.
      channel_id: The id of the Slack channel to send and receive messages.
      model_uri: The URI of the model waiting for human review.

    Returns:
      A single boolean value indicating whether or not this model is accepted.
      It will return True when 'Approve' is received from the channel.
      It will return False when 'Decline' is received from the channel.

    """
    sc = SlackClient(slack_token)
    msg = _NOTIFY_MODEL_REVIEW_TEMPLATE.format(model_uri)
    ts = 0
    if sc.rtm_connect():
      sc.rtm_send_message(channel=channel_id, message=msg)

      while sc.server.connected:
        payload_list = sc.rtm_read()
        if not payload_list:
          continue

        for payload in payload_list:
          if payload.get('ok') and payload.get('reply_to') == 0 and not ts:
            ts = payload['ts']
            continue
          elif payload.get('type') == 'message' and payload.get(
              'channel') == channel_id and payload.get('text') and payload.get(
                  'thread_ts') == ts:
            if payload.get('text').lower() in _APPROVE_TEXT:
              tf.logging.info('User %s approves the model located at %s',
                              payload.get('user'), model_uri)
              return True
            elif payload.get('text').lower() in _DECLINE_TEXT:
              tf.logging.info('User %s declines the model located at %s',
                              payload.get('user'), model_uri)
              return False
            else:
              unrecognized_text = payload.get('text')
              tf.logging.info('Unrecognized response: %s', unrecognized_text)
              sc.rtm_send_message(
                  channel=channel_id,
                  message=_NOTIFY_CORRECT_REPLY_TEMPLATE.format(
                      unrecognized_text),
                  thread=ts)

  def Do(self, input_dict: Dict[Text, List[types.TfxType]],
         output_dict: Dict[Text, List[types.TfxType]],
         exec_properties: Dict[Text, Any]) -> None:
    """Get human review result on a model through Slack channel.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.
      output_dict: Output dict from key to a list of artifacts, including:
        - slack_blessing: model blessing result.
      exec_properties: A dict of execution properties, including:
        - slack_token: Token used to setup connection with slack server.
        - channel_id: The id of the Slack channel to send and receive messages.
        - timeout_sec: How long do we wait for response, in seconds.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    # Fetch execution properties from exec_properties dict.
    slack_token = exec_properties['slack_token']
    channel_id = exec_properties['channel_id']
    timeout_sec = exec_properties['timeout_sec']

    # Fetch input URIs from input_dict.
    model_export_uri = types.get_single_uri(input_dict['model_export'])
    model_blessing_uri = types.get_single_uri(input_dict['model_blessing'])

    # Fetch output artifact from output_dict.
    slack_blessing = types.get_single_instance(output_dict['slack_blessing'])

    # We only consider a model as blessed if both of the following conditions
    # are met:
    # - The model is blessed by model validator. This is determined by looking
    #   for file named 'BLESSED' from the output from Model Validator.
    # - The model is blessed by a human reviewer. This logic is in
    #   _fetch_slack_blessing().
    try:
      with Timeout(timeout_sec):
        blessed = tf.gfile.Exists(os.path.join(
            model_blessing_uri, 'BLESSED')) and self._fetch_slack_blessing(
                slack_token, channel_id, model_export_uri)
    except TimeoutError:  # pylint: disable=undefined-variable
      tf.logging.info('Timeout fetching manual model evaluation result.')
      blessed = False

    # If model is blessed, write an empty file named 'BLESSED' in the assigned
    # output path. Otherwise, write an empty file named 'NOT_BLESSED' instead.
    if blessed:
      io_utils.write_string_file(
          os.path.join(slack_blessing.uri, 'BLESSED'), '')
      slack_blessing.set_int_custom_property('blessed', 1)
    else:
      io_utils.write_string_file(
          os.path.join(slack_blessing.uri, 'NOT_BLESSED'), '')
      slack_blessing.set_int_custom_property('blessed', 0)
    tf.logging.info('Blessing result %s written to %s.', blessed,
                    slack_blessing.uri)
