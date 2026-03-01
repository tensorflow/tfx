# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Utilities to use Vertex Pipelines client."""

import datetime
import time

from absl import logging
from google.cloud.aiplatform import pipeline_jobs
from google.cloud.aiplatform_v1.types import pipeline_state


_PIPELINE_COMPLETE_STATES = frozenset([
    pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED,
    pipeline_state.PipelineState.PIPELINE_STATE_FAILED,
    pipeline_state.PipelineState.PIPELINE_STATE_CANCELLED,
    pipeline_state.PipelineState.PIPELINE_STATE_PAUSED,
])


def poll_job_status(job_id: str, timeout: datetime.timedelta,
                    polling_interval_secs: int):
  """Checks the status of the job.

  NOTE: aiplatform.init() should be already called.

  Args:
    job_id: The relative ID of the pipeline job.
    timeout: Timeout duration for the job execution.
    polling_interval_secs: Interval to check the job status.

  Raises:
    RuntimeError: On (1) unexpected response from service; or (2) on
      unexpected job status; or (2) timed out waiting for finishing.
  """
  deadline = datetime.datetime.now() + timeout
  while datetime.datetime.now() < deadline:
    time.sleep(polling_interval_secs)

    job = pipeline_jobs.PipelineJob.get(resource_name=job_id)
    # '.state' is synced everytime we access the property. So it can change
    # between comparisons. We have to make a copy to compare it multiple times.
    job_state = job.state
    if (job_state ==
        pipeline_state.PipelineState.PIPELINE_STATE_SUCCEEDED):
      logging.info('Job succeeded: %s', job)
      return
    elif job_state in _PIPELINE_COMPLETE_STATES:
      raise RuntimeError('Job is in an unexpected state: %s' % job_state)

  raise RuntimeError('Timed out waiting for job to finish.')
