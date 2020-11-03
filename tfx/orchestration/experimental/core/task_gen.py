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
"""TaskGenerator interface."""

import abc
from typing import List

from tfx.orchestration.experimental.core import task as task_lib


class TaskGenerator(abc.ABC):
  """TaskGenerator interface.

  When their `generate` method is invoked (typically done periodically within an
  orchestration loop), concrete classes implementing this interface are expected
  to generate tasks to execute nodes in a pipeline IR spec or system tasks (eg:
  for garbage collection) based on the state of pipeline execution and related
  details stored in an MLMD db.

  Note on thread safety: Concrete classes of this interface need not have a
  thread-safe implementation. Onus is on the caller to serialize concurrent
  calls to `generate`. Since MLMD db may be updated upon call to `generate`,
  it's also not safe to invoke `generate` concurrently on different instances
  of `TaskGenerator` that refer to the same MLMD db and the same pipeline IR.
  """

  @abc.abstractmethod
  def generate(self) -> List[task_lib.Task]:
    """Generates a list of tasks to be performed.

    Returns:
      A list of `Task`s specifying nodes in a pipeline to be executed or other
      system tasks.
    """
