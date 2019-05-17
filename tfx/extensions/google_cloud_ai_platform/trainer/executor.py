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
"""Helper class to start TFX training jobs on CMLE."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_executor
from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.orchestration.gcp import cmle_runner
from tfx.utils import types

_POLLING_INTERVAL_IN_SECONDS = 30


class Executor(base_executor.BaseExecutor):
  """Start a trainer job on Google Cloud AI Platform (GAIP)."""

  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]):
    """Starts a trainer job on Google Cloud AI Platform.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Trainer.executor.
        custom_config.ai_platform_training_args is consumed by this class.  For
        the full set of parameters supported by Google Cloud AI Platform, refer
        to
        https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs#configuring_the_job

    Returns:
      None
    Raises:
      ValueError: if ai_platform_training_args is not in
      exec_properties.custom_config.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if not exec_properties.get('custom_config',
                               {}).get('ai_platform_training_args'):
      err_msg = '\'ai_platform_training_args\' not found in custom_config.'
      tf.logging.error(err_msg)
      raise ValueError(err_msg)

    training_inputs = exec_properties.get('custom_config',
                                          {}).pop('ai_platform_training_args')
    executor_class_path = '%s.%s' % (tfx_trainer_executor.Executor.__module__,
                                     tfx_trainer_executor.Executor.__name__)
    return cmle_runner.start_cmle_training(input_dict, output_dict,
                                           exec_properties, executor_class_path,
                                           training_inputs)
