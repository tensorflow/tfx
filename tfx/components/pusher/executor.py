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
"""TFX pusher executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import time
from typing import Any, Dict, List, Optional, Text

from absl import logging
import six
import tensorflow as tf

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.proto import pusher_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Aliasing of enum for better readability.
_Versioning = pusher_pb2.Versioning

# Key for model in executor input_dict.
MODEL_KEY = 'model'
# Key for model blessing in executor input_dict.
MODEL_BLESSING_KEY = 'model_blessing'
# Key for infra blessing in executor input_dict.
INFRA_BLESSING_KEY = 'infra_blessing'
# Key for pushed model in executor output_dict.
PUSHED_MODEL_KEY = 'pushed_model'
# Key for PushDestination in exec_properties.
PUSH_DESTINATION_KEY = 'push_destination'
# Key for custom_config in exec_properties.
CUSTOM_CONFIG_KEY = 'custom_config'

# Key for PushedModel artifact properties.
_PUSHED_KEY = 'pushed'
_PUSHED_DESTINATION_KEY = 'pushed_destination'
_PUSHED_VERSION_KEY = 'pushed_version'


# TODO(jjong): Use dataclasses or attrs instead?
class PushResult(object):
  """Data class for push result."""
  __slots__ = ['destination', 'version']

  def __init__(self, destination: Text, version: Optional[Text] = None):
    self.destination = destination
    self.version = version


class BasePusherExecutor(six.with_metaclass(abc.ABCMeta,
                                            base_executor.BaseExecutor)):
  """TFX Pusher executor to push the new TF model to a filesystem target.

  The Pusher component is used to deploy a validated model to a filesystem
  target or serving environment using tf.serving. Pusher depends on the outputs
  of ModelValidator to determine if a model is ready to push. A model is
  considered to be safe to push only if ModelValidator has marked it as BLESSED.
  A push action delivers the model exports produced by Trainer to the
  destination defined in the `push_destination` of the component config.

  To include Pusher in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L104.

  For more details on tf.serving itself, please refer to
  https://tensorflow.org/tfx/guide/pusher. For a tutorial on TF Serving,
  please refer to https://www.tensorflow.org/tfx/guide/serving.
  """

  def _CheckBlessing(
      self, input_dict: Dict[Text, List[types.Artifact]]) -> bool:
    """Check that model is blessed by upstream validators.

    Args:
      input_dict: Input dict from input key to a list of artifacts:
        - model_blessing: A `ModelBlessing` artifact from model validator or
          evaluator.
          Pusher looks for a custom property `blessed` in the artifact to check
          it is safe to push.
        - infra_blessing: An `InfraBlessing` artifact from infra validator.
          Pusher looks for a custom proeprty `blessed` in the artifact to
          determine whether the model is mechanically servable from the model
          server to which Pusher is going to push.

    Returns:
      True if the model is blessed by validator.
    """
    model_blessing = artifact_utils.get_single_instance(
        input_dict[MODEL_BLESSING_KEY])
    # TODO(jyzhao): should this be in driver or executor.
    if not model_utils.is_model_blessed(model_blessing):
      logging.info('Model on %s was not blessed by model validation',
                   model_blessing.uri)
      return False
    if INFRA_BLESSING_KEY in input_dict:
      infra_blessing = artifact_utils.get_single_instance(
          input_dict[INFRA_BLESSING_KEY])
      if not model_utils.is_infra_validated(infra_blessing):
        logging.info('Model on %s was not blessed by infra validator',
                     model_blessing.uri)
        return False
    return True

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Push model to target directory if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator. A push
          action delivers the model exports produced by Trainer to the
          destination defined in component config.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: A dict of execution properties, including:
        - push_destination: JSON string of pusher_pb2.PushDestination instance,
          providing instruction of destination to push model.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    model_push = artifact_utils.get_single_instance(
        output_dict[PUSHED_MODEL_KEY])

    if not self._CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return

    model_export = artifact_utils.get_single_instance(input_dict[MODEL_KEY])

    # Copy Model artifact contents to the PushedModel artifact. Note that
    # although the contents of Model and PushedModel artifacts looks the same,
    # they serve the different purpose, and it is more appropriate to have two
    # copies of the model in each artifact. Some future considerations:
    # 1) Each artifact might have different GC policy in the future, and we
    #    don't want to have dangling deployment due to deletion of Model
    #    artifact (which is a Trainer output).
    # 2) PushedModel artifact might rewrite the input model in the future.
    io_utils.copy_dir(path_utils.serving_model_path(model_export.uri),
                      model_push.uri)

    push_dest = pusher_pb2.PushDestination()
    serialized_push_dest = exec_properties.get(PUSH_DESTINATION_KEY)
    if serialized_push_dest:
      json_format.Parse(serialized_push_dest, push_dest)
    # TODO(jjong): Why is custom_config not serialized?
    custom_config = exec_properties.get(CUSTOM_CONFIG_KEY, {})

    try:
      result = self._PushImpl(
          model_push=model_push,
          push_destination=push_dest,
          custom_config=custom_config,
      )
    except:  # pylint: disable=broad-except, bare-except
      self._MarkNotPushed(model_push)
    else:
      self._MarkPushed(model_push,
                       pushed_destination=result.destination,
                       pushed_version=result.version)

  @abc.abstractmethod
  def _PushImpl(self, model_push: types.Artifact,
                push_destination: pusher_pb2.PushDestination,
                custom_config: Dict[Text, Any]) -> PushResult:
    """Implementation of the push behavior.

    Args:
      model_push: PushedModel artifact that contains the model to be deployed.
      push_destination: Config about push destination.
      custom_config: Custom configuration for custom push destination.

    Returns:
      PushResult.

    Raises:
      NotImplementedError: If given push_destination is not implemented.
    """

  def _MarkPushed(self, model_push: types.Artifact, pushed_destination: Text,
                  pushed_version: Optional[Text] = None) -> None:
    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property(
        _PUSHED_DESTINATION_KEY, pushed_destination)
    if pushed_version is not None:
      model_push.set_string_custom_property(
          _PUSHED_VERSION_KEY, pushed_version)
    logging.info('Model pushed to %s.', model_push.uri)

  def _MarkNotPushed(self, model_push: types.Artifact):
    model_push.set_int_custom_property('pushed', 0)
    logging.info('Model was not pushed.')


class Executor(BasePusherExecutor):
  """Pusher Executor implementation for the filesystem destination."""

  # TODO(jyzhao): support rpc push and verification.
  def _PushImpl(self, model_push: types.Artifact,
                push_destination: pusher_pb2.PushDestination,
                custom_config: Dict[Text, Any]) -> PushResult:
    del custom_config  # Unused.

    dest = push_destination.WhichOneof('destination')
    if dest != 'filesystem':
      raise NotImplementedError('Unsupported push destination {}'.format(dest))

    config = push_destination.filesystem
    if config.versioning == _Versioning.AUTO:
      config.versioning = _Versioning.UNIX_TIMESTAMP
    if config.versioning == _Versioning.UNIX_TIMESTAMP:
      version = str(int(time.time()))
    else:
      raise NotImplementedError(
          'Invalid Versioning {}'.format(config.versioning))
    logging.info('Model version: %s', version)
    serving_path = os.path.join(config.base_directory, version)

    if tf.io.gfile.exists(serving_path):
      # If model is already successfully copied to outside before, stop copying.
      # This is because model validator might blessed same model twice (check
      # mv driver) with different blessing output, we still want Pusher to
      # handle the mv output again to keep metadata tracking, but no need to
      # copy to outside path again..
      logging.info(
          'Destination directory %s already exists, skipping current push.',
          serving_path)
    else:
      # tf.serving won't load partial model, it will retry until fully copied.
      io_utils.copy_dir(model_push.uri, serving_path)
      logging.info('Model written to serving path %s.', serving_path)

    return PushResult(destination=serving_path, version=version)
