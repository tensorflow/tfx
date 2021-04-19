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
"""Task scheduler interface and registry."""

import abc
import typing
from typing import Dict, List, Optional, Type, TypeVar

import attr
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import task as task_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2


@attr.s(auto_attribs=True, frozen=True)
class TaskSchedulerResult:
  """Response from the task scheduler.

  Attributes:
    status: Scheduler status that reflects scheduler level issues, such as task
      cancellation, failure to start the executor, etc. Executor status set in
      `executor_output` matters if the scheduler status is `OK`. Otherwise,
      `executor_output` may be `None` and is ignored.
    executor_output: An instance of `ExecutorOutput` containing the results of
      task execution. Neither or one of `executor_output` or `output_artifacts`
      but not both should be returned in the response.
    output_artifacts: Output artifacts dict containing the results of task
      execution. Neither or one of `executor_output` or `output_artifacts` but
      not both should be returned in the response.
  """
  status: status_lib.Status
  executor_output: Optional[execution_result_pb2.ExecutorOutput] = None
  output_artifacts: Optional[Dict[str, List[types.Artifact]]] = None

  def __attrs_post_init__(self):
    if self.executor_output is not None and self.output_artifacts is not None:
      raise ValueError(
          'Only one of output_artifacts or executor_output must be set.')


class TaskScheduler(abc.ABC):
  """Interface for task schedulers."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: task_lib.Task):
    """Constructor.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR proto.
      task: Task to be executed.
    """
    self.mlmd_handle = mlmd_handle
    self.pipeline = pipeline
    self.task = task

  @abc.abstractmethod
  def schedule(self) -> TaskSchedulerResult:
    """Schedules task execution and returns the results of execution.

    This method blocks until task execution completes (successfully or not) or
    until explicitly cancelled by a call to `cancel`. When cancelled, `schedule`
    is expected to stop any ongoing work, clean up and return as soon as
    possible. Note that `cancel` will be invoked from a different thread than
    `schedule` and hence the concrete implementations must be thread safe. It's
    technically possible for `cancel` to be invoked before `schedule`; scheduler
    implementations should handle this case by returning from `schedule`
    immediately.
    """

  @abc.abstractmethod
  def cancel(self) -> None:
    """Cancels task scheduler.

    This method will be invoked from a different thread than the thread that's
    blocked on call to `schedule`. `cancel` must return immediately when called.
    Upon cancellation, `schedule` method is expected to stop any ongoing work,
    clean up and return as soon as possible. It's technically possible for
    `cancel` to be invoked before `schedule`; scheduler implementations should
    handle this case by returning from `schedule` immediately.
    """


T = TypeVar('T', bound='TaskSchedulerRegistry')


class TaskSchedulerRegistry:
  """A registry for task schedulers."""

  _task_scheduler_registry = {}

  @classmethod
  def register(cls: Type[T], url: str,
               scheduler_class: Type[TaskScheduler]) -> None:
    """Registers a new task scheduler for the given url.

    Args:
      url: The URL associated with the task scheduler. It should either be the
        node type url or executor spec url.
      scheduler_class: The class that will be instantiated for a matching task.

    Raises:
      ValueError: If `url` is already in the registry.
    """
    if url in cls._task_scheduler_registry:
      raise ValueError(f'A task scheduler already exists for the url: {url}')
    cls._task_scheduler_registry[url] = scheduler_class

  @classmethod
  def clear(cls: Type[T]) -> None:
    cls._task_scheduler_registry.clear()

  @classmethod
  def create_task_scheduler(cls: Type[T], mlmd_handle: metadata.Metadata,
                            pipeline: pipeline_pb2.Pipeline,
                            task: task_lib.Task) -> TaskScheduler:
    """Creates a task scheduler for the given task.

    The task is matched as follows:
    1. The node type name of the node associated with the task is looked up in
       the registry and a scheduler is instantiated if present.
    2. Next, the executor spec url of the node (if one exists) is looked up in
       the registry and a scheduler is instantiated if present. This assumes
       deployment_config packed in the pipeline IR is of type
       `IntermediateDeploymentConfig`.
    3. Lastly, a ValueError is raised if no match can be found.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR.
      task: The task that needs to be scheduled.

    Returns:
      An instance of `TaskScheduler` for the given task.

    Raises:
      NotImplementedError: Raised if not an `ExecNodeTask`.
      ValueError: If a scheduler could not be found in the registry for the
        given task.
    """
    if not task_lib.is_exec_node_task(task):
      raise NotImplementedError(
          'Can create a task scheduler only for an `ExecNodeTask`.')
    task = typing.cast(task_lib.ExecNodeTask, task)

    scheduler_class = cls._task_scheduler_registry.get(
        task.get_pipeline_node().node_info.type.name)
    if scheduler_class is not None:
      return scheduler_class(
          mlmd_handle=mlmd_handle, pipeline=pipeline, task=task)

    if not pipeline.deployment_config.Is(
        pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR):
      raise ValueError('No deployment config found in pipeline IR.')
    depl_config = pipeline_pb2.IntermediateDeploymentConfig()
    pipeline.deployment_config.Unpack(depl_config)
    node_id = task.node_uid.node_id
    if node_id not in depl_config.executor_specs:
      raise ValueError(
          'Executor spec for node id `{}` not found in pipeline IR.'.format(
              node_id))
    executor_spec_type_url = depl_config.executor_specs[node_id].type_url
    return cls._task_scheduler_registry[executor_spec_type_url](
        mlmd_handle=mlmd_handle, pipeline=pipeline, task=task)
