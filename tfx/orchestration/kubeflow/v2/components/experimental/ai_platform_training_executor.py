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
"""Executor for AI Platform Training component."""

import datetime
from typing import Any, Dict, List

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.orchestration.launcher import container_common
from tfx.utils import json_utils

_POLLING_INTERVAL_IN_SECONDS = 30
_CONNECTION_ERROR_RETRY_LIMIT = 5

# Keys for AIP training config.
PROJECT_CONFIG_KEY = 'project_id'
TRAINING_JOB_CONFIG_KEY = 'training_job'
JOB_ID_CONFIG_KEY = 'job_id'
LABELS_CONFIG_KEY = 'labels'
CONFIG_KEY = 'aip_training_config'


class AiPlatformTrainingExecutor(base_executor.BaseExecutor):
  """AI Platform Training executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:

    self._log_startup(input_dict, output_dict, exec_properties)

    aip_config = json_utils.loads(exec_properties.pop(CONFIG_KEY))

    assert aip_config, 'AIP training config is not found.'

    training_job = aip_config.pop(TRAINING_JOB_CONFIG_KEY)
    job_id = aip_config.pop(JOB_ID_CONFIG_KEY)
    project = aip_config.pop(PROJECT_CONFIG_KEY)

    # Resolve parameters.
    training_job['training_input'][
        'args'] = container_common._resolve_container_command_line(  # pylint: disable=protected-access
            cmd_args=training_job['training_input']['args'],
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties)

    training_job['job_id'] = job_id or 'tfx_{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    # Invoke CMLE job
    runner._launch_cloud_training(  # pylint: disable=protected-access
        project=project,
        training_job=training_job)
