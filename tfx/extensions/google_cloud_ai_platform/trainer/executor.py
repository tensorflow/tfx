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
"""Helper class to start TFX training jobs on AI Platform."""

from typing import Any, Dict, List

import absl
from tfx import types
from tfx.components.trainer import executor as tfx_trainer_executor
from tfx.dsl.components.base import base_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.constants import ENABLE_VERTEX_KEY
from tfx.extensions.google_cloud_ai_platform.constants import VERTEX_REGION_KEY
from tfx.types import standard_component_specs
from tfx.utils import doc_controls
from tfx.utils import json_utils
from tfx.utils import name_utils

TRAINING_ARGS_KEY = doc_controls.documented(
    obj='ai_platform_training_args',
    doc='Keys to the items in custom_config of Trainer for passing '
    'training_job to AI Platform, and the GCP project under which '
    'the training job will be executed. In Vertex AI, this corresponds to '
    'a CustomJob as defined in:'
    'https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs#CustomJob.'
    'In CAIP, this corresponds to TrainingInputs as defined in:'
    'https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput'
)

JOB_ID_KEY = doc_controls.documented(
    obj='ai_platform_training_job_id',
    doc='Keys to the items in custom_config of Trainer for specifying job id.')

LABELS_KEY = doc_controls.documented(
    obj='ai_platform_training_labels',
    doc='Keys to the items in custom_config of Trainer for specifying labels for '
    'training jobs on the AI Platform only. Not applicable for Vertex AI, where labels '
    'are specified in the CustomJob as defined in: '
    'https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs#CustomJob.'
)

ENABLE_UCAIP_KEY = doc_controls.documented(
    obj='ai_platform_training_enable_ucaip',
    doc='Deprecated. Please use ENABLE_VERTEX_KEY instead. Keys to the items in'
    ' custom_config of Trainer for enabling uCAIP Training. ')

UCAIP_REGION_KEY = doc_controls.documented(
    obj='ai_platform_training_ucaip_region',
    doc='Deprecated. Please use VERTEX_REGION_KEY instead. Keys to the items in'
    ' custom_config of Trainer for specifying the region of uCAIP.')


class GenericExecutor(base_executor.BaseExecutor):
  """Start a trainer job on Google Cloud AI Platform using a generic Trainer."""

  def _GetExecutorClass(self):
    return tfx_trainer_executor.GenericExecutor

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]):
    """Starts a trainer job on Google Cloud AI Platform.

    Args:
      input_dict: Passthrough input dict for tfx.components.Trainer.executor.
      output_dict: Passthrough input dict for tfx.components.Trainer.executor.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Trainer.executor. custom_config.ai_platform_training_args
        and custom_config.ai_platform_training_job_id are consumed by this
        class. For the full set of parameters supported by Vertex AI CustomJob,
        refer to
        https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs#CustomJob
          For the full set of parameters supported by Google Cloud AI Platform
          (CAIP), refer to
        https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs#configuring_the_job

    Returns:
      None
    Raises:
      ValueError: if ai_platform_training_args is not in
      exec_properties.custom_config.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(standard_component_specs.CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')

    job_args = custom_config.get(TRAINING_ARGS_KEY)
    if job_args is None:
      err_msg = '\'%s\' not found in custom_config.' % TRAINING_ARGS_KEY
      absl.logging.error(err_msg)
      raise ValueError(err_msg)

    job_labels = {
        **job_args.get('labels', {}),
        **custom_config.get(LABELS_KEY, {})
    }

    job_id = custom_config.get(JOB_ID_KEY)

    enable_vertex = custom_config.get(
        ENABLE_VERTEX_KEY, custom_config.get(ENABLE_UCAIP_KEY, False))
    vertex_region = custom_config.get(VERTEX_REGION_KEY,
                                      custom_config.get(UCAIP_REGION_KEY))

    executor_class = self._GetExecutorClass()
    executor_class_path = name_utils.get_full_name(executor_class)
    # Note: exec_properties['custom_config'] here is a dict.
    return runner.start_cloud_training(input_dict, output_dict, exec_properties,
                                       executor_class_path, job_args, job_id,
                                       job_labels, enable_vertex, vertex_region)


class Executor(GenericExecutor):
  """Start a trainer job on Google Cloud AI Platform using a default Trainer."""

  def _GetExecutorClass(self):
    return tfx_trainer_executor.Executor
