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
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union

import attr
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib


@attr.s(auto_attribs=True, frozen=True)
class ExecutorNodeOutput:
  """Output of a node containing an executor.

  Attributes:
    executor_output: Output of node execution (if any).
  """
  executor_output: Optional[execution_result_pb2.ExecutorOutput] = None


@attr.s(auto_attribs=True, frozen=True)
class ImporterNodeOutput:
  """Importer system node output.

  Attributes:
    output_artifacts: Output artifacts resulting from importer node execution.
  """
  output_artifacts: Dict[str, List[types.Artifact]]


@attr.s(auto_attribs=True, frozen=True)
class ResolverNodeOutput:
  """Resolver system node output.

  Attributes:
    resolved_input_artifacts: Artifacts resolved by resolver system node.
  """
  resolved_input_artifacts: Dict[str, List[types.Artifact]]


@attr.s(auto_attribs=True, frozen=True)
class TaskSchedulerResult:
  """Response from the task scheduler.

  Attributes:
    status: Scheduler status that reflects scheduler level issues, such as task
      cancellation, failure to start the executor, etc.
    output: Output of task scheduler execution.
  """
  status: status_lib.Status
  output: Union[ExecutorNodeOutput, ImporterNodeOutput,
                ResolverNodeOutput] = ExecutorNodeOutput()


_TaskT = TypeVar('_TaskT', bound=task_lib.Task)


class TaskScheduler(abc.ABC, Generic[_TaskT]):
  """Interface for task schedulers."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: _TaskT):
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
  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    """Cancels task scheduler.

    This method will be invoked from a different thread than the thread that's
    blocked on call to `schedule`. `cancel` must return immediately when called.
    Upon cancellation, `schedule` method is expected to stop any ongoing work,
    clean up and return as soon as possible. It's technically possible for
    `cancel` to be invoked before `schedule`; scheduler implementations should
    handle this case by returning from `schedule` immediately.

    Args:
      cancel_task: The task of this cancellation.
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
      ValueError: If `url` is already in the registry for a different
        scheduler_class.
    """
    if cls._task_scheduler_registry.get(url) not in (None, scheduler_class):
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

    if not isinstance(task, task_lib.ExecNodeTask):
      raise NotImplementedError(
          'Can create a task scheduler only for an `ExecNodeTask`.')

    try:
      scheduler_class = cls._scheduler_class_for_node_type(task)
    except ValueError as e1:
      try:
        scheduler_class = cls._scheduler_class_for_executor_spec(pipeline, task)
      except ValueError as e2:
        raise ValueError(f'No task scheduler found: {e1}, {e2}') from None

    return scheduler_class(
        mlmd_handle=mlmd_handle, pipeline=pipeline, task=task)

  @classmethod
  def _scheduler_class_for_node_type(
      cls: Type[T], task: task_lib.ExecNodeTask) -> Type[TaskScheduler]:
    """Returns scheduler class for node type or raises error if none registered."""
    node_type = task.get_pipeline_node().node_info.type.name
    scheduler_class = cls._task_scheduler_registry.get(node_type)
    if scheduler_class is None:
      raise ValueError(
          f'No task scheduler registered for node type: {node_type}')
    return scheduler_class

  @classmethod
  def _scheduler_class_for_executor_spec(
      cls: Type[T], pipeline: pipeline_pb2.Pipeline,
      task: task_lib.ExecNodeTask) -> Type[TaskScheduler]:
    """Returns scheduler class for executor spec url if feasible, raises error otherwise."""
    if not pipeline.deployment_config.Is(
        pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR):
      raise ValueError('No deployment config found in pipeline IR')
    depl_config = pipeline_pb2.IntermediateDeploymentConfig()
    pipeline.deployment_config.Unpack(depl_config)
    node_id = task.node_uid.node_id
    if node_id not in depl_config.executor_specs:
      raise ValueError(f'Executor spec not found for node id: {node_id}')
    executor_spec_type_url = depl_config.executor_specs[node_id].type_url
    scheduler_class = cls._task_scheduler_registry.get(executor_spec_type_url)
    if scheduler_class is None:
      raise ValueError(
          f'No task scheduler registered for executor spec type url: '
          f'{executor_spec_type_url}')
    return scheduler_class
