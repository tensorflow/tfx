# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Helper class to start TFX multi-worker training jobs on GKE."""

from typing import Any, Dict, List, Text

import absl

from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.extensions.google_cloud_kubernetes import runner
from tfx.orchestration import test_utils
from tfx.utils import json_utils

# Keys to the items in custom_config passed as a part of exec_properties.
TRAINING_ARGS_KEY = 'gke_training_args'
_CUSTOM_CONFIG_KEY = 'custom_config'


class GenericExecutor(base_executor.BaseExecutor):
  """Start a trainer job on Google Kubernetes Engine using a generic Trainer."""

  def _GetExecutorClass(self):
    return tfx_trainer_executor.GenericExecutor

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]):
    """Starts a trainer job on Google Kubernetes Engine.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Trainer.executor. custom_config.gke_training_args
        is consumed by this class.

    Returns:
      None
    Raises:
      ValueError: if gke_training_args is not in exec_properties.custom_config.
      RuntimeError: if the Google Kubernetes Engine training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')

    training_inputs = custom_config.get(TRAINING_ARGS_KEY)
    if training_inputs is None:
      err_msg = '\'%s\' not found in custom_config.' % TRAINING_ARGS_KEY
      absl.logging.error(err_msg)
      raise ValueError(err_msg)

    executor_class = self._GetExecutorClass()
    executor_class_path = '%s.%s' % (executor_class.__module__,
                                     executor_class.__name__)

    if self._context is not None and self._context.unique_id is not None:
      unique_id = str(self._context.unique_id)
    else:
      absl.logging.warning(
          "Missing unique_id in executor, using a random id instead.")
      unique_id = test_utils.random_id()

    # Note: exec_properties['custom_config'] here is a dict.
    return runner.start_gke_training(input_dict, output_dict, exec_properties,
                                     executor_class_path, training_inputs,
                                     unique_id)


class Executor(GenericExecutor):
  """Start a trainer job on Google Kubernetes Engine using a default Trainer."""

  def _GetExecutorClass(self):
    return tfx_trainer_executor.Executor
