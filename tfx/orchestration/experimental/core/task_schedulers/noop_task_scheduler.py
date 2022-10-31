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
"""A no-op task scheduler to aid in testing."""

from absl import logging

from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.proto.orchestration import execution_result_pb2
from tfx.utils import status as status_lib


class NoOpTaskScheduler(ts.TaskScheduler[task_lib.ExecNodeTask]):
  """A no-op task scheduler to aid in testing."""

  def schedule(self) -> ts.TaskSchedulerResult:
    logging.info('Processing ExecNodeTask: %s', self.task)
    executor_output = execution_result_pb2.ExecutorOutput()
    executor_output.execution_result.code = status_lib.Code.OK
    for key, artifacts in self.task.output_artifacts.items():
      for artifact in artifacts:
        executor_output.output_artifacts[key].artifacts.add().CopyFrom(
            artifact.mlmd_artifact)
    result = ts.TaskSchedulerResult(
        status=status_lib.Status(code=status_lib.Code.OK),
        output=ts.ExecutorNodeOutput(executor_output=executor_output))
    logging.info('Result: %s', result)
    return result

  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    pass
