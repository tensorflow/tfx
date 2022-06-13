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
"""A task scheduler for Resolver system node."""

from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.utils import status as status_lib


class ResolverTaskScheduler(task_scheduler.TaskScheduler[task_lib.ExecNodeTask]
                           ):
  """A task scheduler for Resolver system node."""

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    return task_scheduler.TaskSchedulerResult(
        status=status_lib.Status(code=status_lib.Code.OK),
        output=task_scheduler.ResolverNodeOutput(
            resolved_input_artifacts=self.task.input_artifacts))

  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    pass
