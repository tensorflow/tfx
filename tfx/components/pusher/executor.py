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

import os
import time
from typing import Any, Dict, List, Optional, Text

from absl import logging
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

# Key for PushedModel artifact properties.
_PUSHED_KEY = 'pushed'
_PUSHED_DESTINATION_KEY = 'pushed_destination'
_PUSHED_VERSION_KEY = 'pushed_version'


class Executor(base_executor.BaseExecutor):
  """TFX Pusher executor to push the new TF model to a filesystem target.

  The Pusher component is used to deploy a validated model to a filesystem
  target or serving environment using tf.serving.  Pusher depends on the outputs
  of ModelValidator to determine if a model is ready to push. A model is
  considered to be safe to push only if ModelValidator has marked it as BLESSED.
  A push action delivers the model exports produced by Trainer to the
  destination defined in the ``push_destination`` of the component config.

  To include Pusher in a TFX pipeline, configure your pipeline similar to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L104.

  For more details on tf.serving itself, please refer to
  https://tensorflow.org/tfx/guide/pusher.  For a tutuorial on TF Serving,
  please refer to https://www.tensorflow.org/tfx/guide/serving.
  """

  def CheckBlessing(self, input_dict: Dict[Text, List[types.Artifact]]) -> bool:
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
    # TODO(jyzhao): should this be in driver or executor.
    maybe_model_blessing = input_dict.get(MODEL_BLESSING_KEY)
    if maybe_model_blessing:
      model_blessing = artifact_utils.get_single_instance(maybe_model_blessing)
      if not model_utils.is_model_blessed(model_blessing):
        logging.info('Model on %s was not blessed by model validation',
                     model_blessing.uri)
        return False
    maybe_infra_blessing = input_dict.get(INFRA_BLESSING_KEY)
    if maybe_infra_blessing:
      infra_blessing = artifact_utils.get_single_instance(maybe_infra_blessing)
      if not model_utils.is_infra_validated(infra_blessing):
        logging.info('Model on %s was not blessed by infra validator',
                     model_blessing.uri)
        return False
    if not maybe_model_blessing and not maybe_infra_blessing:
      logging.warning('Pusher is going to push the model without validation. '
                      'Consider using Evaluator or InfraValidator in your '
                      'pipeline.')
    return True

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Push model to target directory if blessed.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.  A push
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
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return
    model_export = artifact_utils.get_single_instance(input_dict[MODEL_KEY])
    model_path = path_utils.serving_model_path(model_export.uri)

    # Push model to the destination, which can be listened by a model server.
    #
    # If model is already successfully copied to outside before, stop copying.
    # This is because model validator might blessed same model twice (check
    # mv driver) with different blessing output, we still want Pusher to
    # handle the mv output again to keep metadata tracking, but no need to
    # copy to outside path again..
    # TODO(jyzhao): support rpc push and verification.
    push_destination = pusher_pb2.PushDestination()
    json_format.Parse(exec_properties['push_destination'], push_destination)

    destination_kind = push_destination.WhichOneof('destination')
    if destination_kind == 'filesystem':
      fs_config = push_destination.filesystem
      if fs_config.versioning == _Versioning.AUTO:
        fs_config.versioning = _Versioning.UNIX_TIMESTAMP
      if fs_config.versioning == _Versioning.UNIX_TIMESTAMP:
        model_version = str(int(time.time()))
      else:
        raise NotImplementedError(
            'Invalid Versioning {}'.format(fs_config.versioning))
      logging.info('Model version: %s', model_version)
      serving_path = os.path.join(fs_config.base_directory, model_version)

      if tf.io.gfile.exists(serving_path):
        logging.info(
            'Destination directory %s already exists, skipping current push.',
            serving_path)
      else:
        # tf.serving won't load partial model, it will retry until fully copied.
        io_utils.copy_dir(model_path, serving_path)
        logging.info('Model written to serving path %s.', serving_path)
    else:
      raise NotImplementedError(
          'Invalid push destination {}'.format(destination_kind))

    # Copy the model to pushing uri for archiving.
    io_utils.copy_dir(model_path, model_push.uri)
    self._MarkPushed(model_push,
                     pushed_destination=serving_path,
                     pushed_version=model_version)
    logging.info('Model pushed to %s.', model_push.uri)

  def _MarkPushed(self, model_push: types.Artifact, pushed_destination: Text,
                  pushed_version: Optional[Text] = None) -> None:
    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property(
        _PUSHED_DESTINATION_KEY, pushed_destination)
    if pushed_version is not None:
      model_push.set_string_custom_property(_PUSHED_VERSION_KEY, pushed_version)

  def _MarkNotPushed(self, model_push: types.Artifact):
    model_push.set_int_custom_property('pushed', 0)
