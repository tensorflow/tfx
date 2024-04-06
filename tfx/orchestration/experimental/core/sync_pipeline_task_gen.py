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
"""TaskGenerator implementation for sync pipelines."""

import collections
import textwrap
from typing import Callable, Dict, List, Mapping, Optional, Set

from absl import logging
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib
from tfx.utils import topsort

from ml_metadata.proto import metadata_store_pb2


_LAZY_TRIGGER_STRATEGIES = frozenset({
    pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED,
    pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_COMPLETED,
})

_UPSTREAM_SUCCESS_OPTIONAL_STRATEGIES = frozenset({
    pipeline_pb2.NodeExecutionOptions.ALL_UPSTREAM_NODES_COMPLETED,
    pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_COMPLETED,
})


class SyncPipelineTaskGenerator(task_gen.TaskGenerator):
  """Task generator for executing a sync pipeline.

  Calling `generate` is not thread-safe. Concurrent calls to `generate` should
  be explicitly serialized. Since MLMD may be updated upon call to `generate`,
  it's also not safe to call `generate` on different instances of this class
  where the instances refer to the same MLMD db and the same pipeline IR.
  """

  def __init__(self,
               mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager,
               fail_fast: bool = False):
    """Constructs `SyncPipelineTaskGenerator`.

    Args:
      mlmd_connection_manager: A `MLMDConnectionManager` instance to manager
        multiple mlmd connections.
      is_task_id_tracked_fn: A callable that returns `True` if a task_id is
        tracked by the task queue.
      service_job_manager: Used for handling service nodes in the pipeline.
      fail_fast: If `True`, pipeline run is aborted immediately if any node
        fails. If `False`, pipeline run is only aborted when no further progress
        can be made due to node failures.
    """
    self._mlmd_connection_manager = mlmd_connection_manager
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager
    self._fail_fast = fail_fast

  def generate(self,
               pipeline_state: pstate.PipelineState) -> List[task_lib.Task]:
    """Generates tasks for executing the next executable nodes in the pipeline.

    The returned tasks must have `exec_task` populated. List may be empty if
    no nodes are ready for execution.

    Args:
      pipeline_state: The `PipelineState` object associated with the pipeline
        for which to generate tasks.

    Returns:
      A `list` of tasks to execute.
    """
    return _Generator(self._mlmd_connection_manager, pipeline_state,
                      self._is_task_id_tracked_fn, self._service_job_manager,
                      self._fail_fast)()

  def get_tasks_for_node(
      self,
      node: node_proto_view.NodeProtoView,
      pipeline_state: pstate.PipelineState,
  ) -> List[task_lib.Task]:
    return _Generator(
        self._mlmd_connection_manager,
        pipeline_state,
        self._is_task_id_tracked_fn,
        self._service_job_manager,
        self._fail_fast,
    ).generate_tasks_for_node(node)


class _Generator:
  """Generator implementation class for SyncPipelineTaskGenerator."""

  def __init__(self,
               mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
               pipeline_state: pstate.PipelineState,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager,
               fail_fast: bool = False):
    self._mlmd_connection_manager = mlmd_connection_manager
    self._mlmd_handle = mlmd_connection_manager.primary_mlmd_handle
    pipeline = pipeline_state.pipeline
    if pipeline.execution_mode != pipeline_pb2.Pipeline.ExecutionMode.SYNC:
      raise ValueError(
          'SyncPipelineTaskGenerator should be instantiated with a pipeline '
          'proto having execution_mode `SYNC`, not `{}`'.format(
              pipeline.execution_mode))
    self._pipeline_state = pipeline_state
    with self._pipeline_state:
      self._node_state_by_node_uid = self._pipeline_state.get_node_states_dict()
    self._pipeline = pipeline
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager
    self._fail_fast = fail_fast
    self._node_proto_view_by_node_id: collections.OrderedDict[
        str, node_proto_view.NodeProtoView
    ] = collections.OrderedDict()

  def generate_tasks_for_node(
      self, node: node_proto_view.NodeProtoView
  ) -> List[task_lib.Task]:
    logging.info('in generate_tasks_for_node')
    return self._generate_tasks_from_resolved_inputs(node)

  def __call__(self) -> List[task_lib.Task]:
    layers = _topsorted_layers(self._pipeline)
    exec_node_tasks = []
    update_node_state_tasks = []
    successful_node_ids = set()
    failed_nodes_dict: Dict[str, status_lib.Status] = {}
    finalize_pipeline_task = None
    lazily_evaluated_node_ids = set()

    # Loop over all nodes before deciding scheduling so we have full knowledge
    # of all the completed/lazy nodes.
    for layer in layers:
      for node in layer:
        node_id = node.node_info.id
        node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
        node_state = self._node_state_by_node_uid[node_uid]
        self._node_proto_view_by_node_id[node_id] = node

        if node.execution_options.strategy in _LAZY_TRIGGER_STRATEGIES:
          lazily_evaluated_node_ids.add(node.node_info.id)
        if node_state.is_success() or (
            node_state.is_failure()
            and node.execution_options.node_success_optional
        ):
          successful_node_ids.add(node_id)
        elif node_state.is_failure():
          failed_nodes_dict[node_id] = node_state.status

    # Collect nodes that cannot be run because they have a failed ancestor.
    unrunnable_node_ids = _unrunnable_nodes(
        self._node_proto_view_by_node_id,
        set(failed_nodes_dict.keys()),
    )

    for layer_nodes in layers:
      for node in layer_nodes:
        node_id = node.node_info.id
        if node_id in successful_node_ids:
          continue
        if node_id in failed_nodes_dict:
          continue
        if not self._trigger_strategy_satisfied(
            node,
            successful_node_ids,
            failed_nodes_dict,
            lazily_evaluated_node_ids,
            unrunnable_node_ids
        ):
          continue
        logging.info(
            '[SyncPipelineTaskGenerator._generate_tasks_for_node] generating'
            ' tasks for node %s',
            node.node_info.id,
        )
        tasks = self._generate_tasks_for_node(node)
        logging.info(
            '[SyncPipelineTaskGenerator._generate_tasks_for_node] generated'
            ' tasks for node %s: %s',
            node.node_info.id,
            [t.task_id for t in tasks],
        )
        for task in tasks:
          if isinstance(task, task_lib.UpdateNodeStateTask):
            if pstate.is_node_state_success(
                task.state) or (pstate.is_node_state_failure(task.state) and
                                node.execution_options.node_success_optional):
              successful_node_ids.add(node_id)
            elif pstate.is_node_state_failure(task.state):
              failed_nodes_dict[node_id] = task.status
              # While the pipeline can still proceed depending on the trigger
              # strategy of descending nodes, the fail fast option should only
              # be used together with ALL_UPSTREAM_NODES_SUCCEEDED since it will
              # fail the pipeline if any node fails.
              if self._fail_fast:
                finalize_pipeline_task = self._abort_task(failed_nodes_dict)
            update_node_state_tasks.append(task)
          elif isinstance(task, task_lib.ExecNodeTask):
            exec_node_tasks.append(task)

        # TODO(b/308161293): Remove this and check for updates in later layers
        # as well.
        if finalize_pipeline_task:
          break
      if finalize_pipeline_task:
        break

    # Always update node states if possible.
    result = update_node_state_tasks
    # If finalize_pipeline_task is set here then we should be in fail_fast
    # mode. Will only update node states and finalize pipeline, ignoring other
    # tasks.
    if finalize_pipeline_task:
      result.append(finalize_pipeline_task)
      return result

    # Because we can find newly failed nodes from UpdateNodeStateTask
    # recompute all unrunnable nodes so we can fail the pipeline in this
    # loop.
    # Note that because we only ever append to failed_nodes_dict this set
    # is guaranteed to contain at least the unrunnable nodes we originally
    # computed.
    unrunnable_node_ids = _unrunnable_nodes(
        self._node_proto_view_by_node_id,
        set(failed_nodes_dict.keys()),
    )

    # Nodes that are still runnable have neither succeeded nor failed, don't
    # have a failed ancestor, or have a triggering strategy that ignores
    # upstream failures.
    runnable_node_ids = self._node_proto_view_by_node_id.keys() - (
        unrunnable_node_ids
        | successful_node_ids
        | failed_nodes_dict.keys()
    )

    # If there are no more runnable nodes, then we finalize the pipeline,
    # otherwise run our exec_node tasks,
    if not runnable_node_ids:
      logging.info(
          'No more runnable nodes in pipeline, finalizing. Successful nodes:'
          ' %s, failed nodes: %s, unrunnable nodes: %s.',
          successful_node_ids,
          failed_nodes_dict.keys(),
          unrunnable_node_ids,
      )
      if failed_nodes_dict:
        result.append(self._abort_task(failed_nodes_dict))
      else:
        result.append(
            task_lib.FinalizePipelineTask(
                pipeline_uid=self._pipeline_state.pipeline_uid,
                status=status_lib.Status(code=status_lib.Code.OK),
            )
        )
    else:
      result.extend(exec_node_tasks)

    return result

  def _generate_tasks_for_node(
      self, node: node_proto_view.NodeProtoView) -> List[task_lib.Task]:
    """Generates list of tasks for the given node."""
    node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
    node_id = node.node_info.id
    result = []

    node_state = self._node_state_by_node_uid[node_uid]
    if node_state.state in (
        pstate.NodeState.STOPPING,
        pstate.NodeState.STOPPED,
    ):
      logging.info('Ignoring node in state \'%s\' for task generation: %s',
                   node_state.state, node_uid)
      return result

    # If this is a pure service node, there is no ExecNodeTask to generate
    # but we ensure node services and check service status.
    service_status = self._ensure_node_services_if_pure(node_id)
    if service_status is not None:
      if service_status.code == service_jobs.ServiceStatusCode.FAILED:
        # TODO(b/205642811): Mark all pending executions as either failed (if
        # active) or canceled (if new), and delete the the executions temporary
        # and output directories.
        error_msg = f'service job failed; error message: {service_status.msg}'
        result.append(
            self._update_node_state_to_failed_task(
                node_uid,
                error_code=status_lib.Code.UNKNOWN,
                error_msg=error_msg,
            )
        )
      elif service_status.code == service_jobs.ServiceStatusCode.SUCCESS:
        logging.info('Service node successful: %s', node_uid)
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid, state=pstate.NodeState.COMPLETE))
      elif (
          service_status.code == service_jobs.ServiceStatusCode.RUNNING
          and node_state.state != pstate.NodeState.RUNNING
      ):
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid, state=pstate.NodeState.RUNNING))
      return result

    # For mixed service nodes, we ensure node services and check service
    # status; pipeline is aborted if the service jobs have failed.
    service_status = self._ensure_node_services_if_mixed(node.node_info.id)
    if service_status:
      if service_status.code == service_jobs.ServiceStatusCode.FAILED:
        error_msg = (
            f'associated service job failed; node uid: {node_uid}, error'
            f' message: {service_status.msg}'
        )
        result.append(
            self._update_node_state_to_failed_task(
                node_uid,
                error_code=status_lib.Code.UNKNOWN,
                error_msg=error_msg,
            )
        )
        return result

    # If a task for the node is already tracked by the task queue, it need
    # not be considered for generation again.
    if self._is_task_id_tracked_fn(
        task_lib.exec_node_task_id_from_node(self._pipeline, node)):
      return result

    node_executions = task_gen_utils.get_executions(self._mlmd_handle, node)
    latest_executions_set = task_gen_utils.get_latest_executions_set(
        node_executions)
    logging.info('latest executions set: %s', latest_executions_set)
    # Generates tasks from resolved inputs if the node doesn't have any
    # execution.
    if not latest_executions_set:
      result.extend(self._generate_tasks_from_resolved_inputs(node))
      return result

    # If all the executions are successful, the node is COMPLETE.
    if all(
        execution_lib.is_execution_successful(e) for e in latest_executions_set
    ):
      logging.info('Node successful: %s', node_uid)
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid, state=pstate.NodeState.COMPLETE))
      return result

    failed_executions = [
        e for e in latest_executions_set if execution_lib.is_execution_failed(e)
    ]
    canceled_executions = [
        e
        for e in latest_executions_set
        if execution_lib.is_execution_canceled(e)
    ]
    if failed_executions:
      if len(failed_executions) > 1:
        error_msg = (f'node {node_uid} failed; error: More than one failed '
                     'executions found in the latest execution set.')
        result.append(
            self._update_node_state_to_failed_task(
                node_uid,
                error_code=status_lib.Code.INTERNAL,
                error_msg=error_msg,
            )
        )
      # If the node has a failed execution, try to retry the failed execution.
      # Retry if under retry limit or if STARTED. STARTED is set upstream
      # so we should respect it here. See b/277257906.
      elif (
          node.execution_options.HasField('max_execution_retries')
          and node.execution_options.max_execution_retries
          >= task_gen_utils.get_num_of_failures_from_failed_execution(
              node_executions, failed_executions[0]
          )
      ) or node_state.state == pstate.NodeState.STARTED:
        retry_executions = (
            task_gen_utils.register_executions_from_existing_executions(
                self._mlmd_handle,
                self._pipeline,
                node,
                failed_executions + canceled_executions,
            )
        )
        result.extend(
            self._generate_tasks_from_existing_execution(
                retry_executions[0], node
            )
        )
      else:
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.FAILED,
                status=task_gen_utils.interpret_status_from_failed_execution(
                    failed_executions[0]
                ),
            )
        )
      return result

    # Restarts canceled node, if the node state is STARTED.
    logging.info('canceled executions: %s', canceled_executions)
    if canceled_executions and node_state.state == pstate.NodeState.STARTED:
      logging.info('restarting node %s', node.node_info.id)
      new_executions = (
          task_gen_utils.register_executions_from_existing_executions(
              self._mlmd_handle, self._pipeline, node, canceled_executions
          )
      )
      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=self._mlmd_handle, execution_id=new_executions[0].id
      ) as execution:
        execution.last_known_state = metadata_store_pb2.Execution.RUNNING

      result.extend(
          self._generate_tasks_from_existing_execution(new_executions[0], node)
      )
      return result

    # If the node has active executions, creates tasks from the oldest active
    # execution.
    oldest_active_execution = next((e for e in latest_executions_set
                                    if execution_lib.is_execution_active(e)),
                                   None)
    if oldest_active_execution:
      result.extend(
          self._generate_tasks_from_existing_execution(oldest_active_execution,
                                                       node))
      return result

    raise RuntimeError('Task generation process should not reach this point.')

  def _update_node_state_to_failed_task(
      self,
      node_uid: task_lib.NodeUid,
      error_code: int,
      error_msg: str,
  ) -> task_lib.Task:
    """Generates fail tasks for a node."""
    error_msg = textwrap.shorten(error_msg, width=512)
    return task_lib.UpdateNodeStateTask(
        node_uid=node_uid,
        state=pstate.NodeState.FAILED,
        status=status_lib.Status(code=error_code, message=error_msg),
    )

  def _generate_tasks_from_existing_execution(
      self, execution: metadata_store_pb2.Execution,
      node: node_proto_view.NodeProtoView) -> List[task_lib.Task]:
    """Generates tasks for a node from its existing execution."""
    logging.info(
        'Generating tasks from existing execution for node: %s',
        node.node_info.id,
    )
    tasks = []
    node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
    with mlmd_state.mlmd_execution_atomic_op(
        mlmd_handle=self._mlmd_handle, execution_id=execution.id) as e:
      e.last_known_state = metadata_store_pb2.Execution.RUNNING

    tasks.append(
        task_lib.UpdateNodeStateTask(
            node_uid=node_uid, state=pstate.NodeState.RUNNING))
    tasks.append(
        task_gen_utils.generate_task_from_execution(self._mlmd_handle,
                                                    self._pipeline, node, e))
    return tasks

  def _generate_tasks_from_resolved_inputs(
      self,
      node: node_proto_view.NodeProtoView,
  ) -> List[task_lib.Task]:
    """Generates tasks for a node by freshly resolving inputs."""
    logging.info(
        'Generating tasks from resolved inputs for node: %s', node.node_info.id
    )
    result = []
    node_uid = task_lib.NodeUid.from_node(self._pipeline, node)

    try:
      resolved_info = task_gen_utils.generate_resolved_info(
          self._mlmd_connection_manager, node, self._pipeline
      )
      logging.info('Resolved inputs: %s', resolved_info)
    except exceptions.InputResolutionError as e:
      error_msg = (f'failure to resolve inputs; node uid: {node_uid}; '
                   f'error: {e.__cause__ or e}')
      result.append(
          self._update_node_state_to_failed_task(
              node_uid, error_code=e.grpc_code_value, error_msg=error_msg
          )
      )
      return result

    if not resolved_info.input_and_params:
      logging.info('Node skipped: %s', node_uid)
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid,
              state=pstate.NodeState.SKIPPED,
              status=status_lib.Status(
                  code=status_lib.Code.OK,
                  message=(
                      'Node execution skipped either due to conditional'
                      ' evaluated to false or no inputs resolved. Please check'
                      ' whether the output of the upstream node was generated'
                      ' successfully.'
                  ),
              ),
          )
      )
      return result

    # Copys artifact types of the external artifacts to local db, in idempotent
    # manner. Idempotency is guaranteed by the artifact type name.
    # The external artifacts will be copies to local db when we register
    # executions. Idempotency is guaranteed by external_id.
    updated_external_artifacts = []
    for input_and_params in resolved_info.input_and_params:
      for artifacts in input_and_params.input_artifacts.values():
        updated_external_artifacts.extend(
            task_gen_utils.update_external_artifact_type(
                self._mlmd_handle, artifacts
            )
        )
    if updated_external_artifacts:
      logging.info(
          'Updated external artifact ids: %s',
          [a.id for a in updated_external_artifacts],
      )

    executions = task_gen_utils.register_executions(
        metadata_handle=self._mlmd_handle,
        execution_type=node.node_info.type,
        contexts=resolved_info.contexts,
        input_and_params=resolved_info.input_and_params,
    )

    result.extend(
        task_gen_utils.generate_tasks_from_one_input(
            metadata_handle=self._mlmd_handle,
            node=node,
            execution=executions[0],
            input_and_param=resolved_info.input_and_params[0],
            contexts=resolved_info.contexts,
            pipeline=self._pipeline,
            execution_node_state=pstate.NodeState.RUNNING,
        )
    )
    return result

  def _ensure_node_services_if_pure(
      self, node_id: str) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is pure service node."""
    if self._service_job_manager.is_pure_service_node(self._pipeline_state,
                                                      node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id)
    return None

  def _ensure_node_services_if_mixed(
      self, node_id: str) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is mixed service node."""
    if self._service_job_manager.is_mixed_service_node(self._pipeline_state,
                                                       node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id)
    return None

  def _upstream_nodes_successful(self, node: node_proto_view.NodeProtoView,
                                 successful_node_ids: Set[str]) -> bool:
    """Returns `True` if all the upstream nodes have been successfully executed."""
    return set(node.upstream_nodes) <= successful_node_ids

  def _upstream_nodes_completed(
      self, node: node_proto_view.NodeProtoView, successful_node_ids: Set[str],
      failed_nodes_dict: Dict[str, status_lib.Status]) -> bool:
    """Returns `True` if all the upstream nodes have been executed or skipped."""
    return set(node.upstream_nodes) <= (
        successful_node_ids | failed_nodes_dict.keys())

  def _lifetime_end_when_subgraph_cannot_progress(
      self,
      node: node_proto_view.NodeProtoView,
      successful_node_ids: Set[str],
      unrunnable_node_ids: Set[str],
      failed_nodes_dict: Mapping[str, status_lib.Status],
  ) -> bool:
    """Returns `True` if all upstream nodes are either COMPLETE or unrunnable."""
    if not (
        start_node := node.execution_options.resource_lifetime.lifetime_start
    ):
      raise ValueError(
          f'Node {node.node_info.id} has trigger strategy'
          ' LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS but no lifetime_start.'
      )
    # If the start node was not successful we will never trigger the end node.
    if start_node not in successful_node_ids:
      return False

    # Otherwise, the end node should run if none of its upstream nodes are
    # runnable.

    # All nodes not in this set are runnable.
    complete_or_unrunnable_nodes = (
        successful_node_ids | unrunnable_node_ids | failed_nodes_dict.keys()
    )

    # Any potentially runnable upstream nodes are the upstream nodes that are
    # not complete or unrunnable.
    runnable_upstream_node_ids = (
        set(node.upstream_nodes) - complete_or_unrunnable_nodes
    )
    logging.info(
        '[LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS trigger check]'
        ' for node %s,'
        ' complete_or_unrunnable nodes: %s, runnable upstream nodes: %s',
        node.node_info.id,
        complete_or_unrunnable_nodes,
        runnable_upstream_node_ids,
    )
    # If this set is empty then the end node should run, otherwise it needs to
    # wait.
    return not runnable_upstream_node_ids

  def _trigger_strategy_satisfied(
      self,
      node: node_proto_view.NodeProtoView,
      successful_node_ids: Set[str],
      failed_nodes_dict: Dict[str, status_lib.Status],
      lazily_evaluated_node_ids: Set[str],
      unrunnable_node_ids: Set[str],
  ) -> bool:
    """Returns `True` if the node's Trigger Strategy is satisfied."""
    if node.execution_options.strategy in _UPSTREAM_SUCCESS_OPTIONAL_STRATEGIES:
      node_trigger_strategy_satisfied = self._upstream_nodes_completed(
          node, successful_node_ids, failed_nodes_dict
      )
    elif node.execution_options.strategy in (
        pipeline_pb2.NodeExecutionOptions.TRIGGER_STRATEGY_UNSPECIFIED,
        pipeline_pb2.NodeExecutionOptions.ALL_UPSTREAM_NODES_SUCCEEDED,
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED,
    ):
      node_trigger_strategy_satisfied = self._upstream_nodes_successful(
          node, successful_node_ids
      )
    elif (
        node.execution_options.strategy
        == pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS
    ):
      node_trigger_strategy_satisfied = (
          self._lifetime_end_when_subgraph_cannot_progress(
              node, successful_node_ids, unrunnable_node_ids, failed_nodes_dict
          )
      )
    else:
      raise NotImplementedError(
          'Unrecognized node triggering strategy: %s' %
          pipeline_pb2.NodeExecutionOptions.TriggerStrategy.Name(
              node.execution_options.strategy))

    if not node_trigger_strategy_satisfied:
      return node_trigger_strategy_satisfied

    # Only check that downstream nodes are otherwise satisfied if there are any
    # downstream nodes, otherwise we should just treat the node as normal.
    if (
        node.execution_options.strategy in _LAZY_TRIGGER_STRATEGIES
        and node.downstream_nodes
    ):
      any_downstream_node_otherwise_ready = False
      successful_or_lazy_node_ids = (
          successful_node_ids | lazily_evaluated_node_ids
      )
      for downstream_node in node.downstream_nodes:
        downstream_trigger = self._trigger_strategy_satisfied(
            self._node_proto_view_by_node_id[downstream_node],
            successful_or_lazy_node_ids,
            failed_nodes_dict,
            lazily_evaluated_node_ids,
            unrunnable_node_ids
        )
        any_downstream_node_otherwise_ready |= downstream_trigger
        if any_downstream_node_otherwise_ready:
          break
      node_trigger_strategy_satisfied &= any_downstream_node_otherwise_ready
    return node_trigger_strategy_satisfied

  def _abort_task(
      self, failed_nodes_dict: Mapping[str, status_lib.Status]
  ) -> task_lib.FinalizePipelineTask:
    """Returns task to abort pipeline execution."""
    logging.error(
        'Pipeline failed due to node failures. Failed nodes:\n%s',
        '\n'.join(
            f'node_id: {node_id}, status: {status}'
            for node_id, status in failed_nodes_dict.items()
        ),
    )
    return task_lib.FinalizePipelineTask(
        pipeline_uid=self._pipeline_state.pipeline_uid,
        status=next(iter(failed_nodes_dict.values())),
    )


def _skipped_node_ids(
    node_states_dict: Dict[task_lib.NodeUid, pstate.NodeState]
) -> Set[str]:
  """Returns the nodes that are marked as skipped in partial run or by user."""
  skipped_node_ids = set()
  for node_uid, node_state in node_states_dict.items():
    if node_state.state in (
        pstate.NodeState.SKIPPED,
        pstate.NodeState.SKIPPED_PARTIAL_RUN,
    ):
      skipped_node_ids.add(node_uid.node_id)
  return skipped_node_ids


def _topsorted_layers(
    pipeline: pipeline_pb2.Pipeline
) -> List[List[node_proto_view.NodeProtoView]]:
  """Returns pipeline nodes in topologically sorted layers."""
  node_by_id = _node_by_id(pipeline)
  return topsort.topsorted_layers(
      [node_proto_view.get_view(node) for node in pipeline.nodes],
      get_node_id_fn=lambda node: node.node_info.id,
      get_parent_nodes=(
          lambda node: [node_by_id[n] for n in node.upstream_nodes]),
      get_child_nodes=(
          lambda node: [node_by_id[n] for n in node.downstream_nodes]))


def _node_by_id(
    pipeline: pipeline_pb2.Pipeline
) -> Dict[str, node_proto_view.NodeProtoView]:
  result = {}
  for node in pipeline.nodes:
    view = node_proto_view.get_view(node)
    result[view.node_info.id] = view
  return result


def _unrunnable_nodes(
    node_by_id: collections.OrderedDict[str, node_proto_view.NodeProtoView],
    failed_node_ids: Set[str],
) -> Set[str]:
  """Returns node_ids of all unrunnable descendant nodes for each member of the given failed_node_ids set."""

  unrunnable = set()
  queue = collections.deque()

  for failed_node_id in failed_node_ids:
    for node_with_upstream_failure in node_by_id[
        failed_node_id
    ].downstream_nodes:
      # Nodes with a upstream success optional trigger strategy can make
      # progress despite a failed upstream node.
      if (
          node_by_id[node_with_upstream_failure].execution_options.strategy
          not in _UPSTREAM_SUCCESS_OPTIONAL_STRATEGIES
      ):
        queue.append(node_with_upstream_failure)

  while queue:
    q_node_id = queue.popleft()
    node = node_by_id[q_node_id]
    start_node = node.execution_options.resource_lifetime.lifetime_start
    if (
        node.execution_options.strategy
        == pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS
        and not (start_node in failed_node_ids or start_node in unrunnable)
    ):
      logging.info(
          '%s is an end node that may still be run since its start node %s'
          ' was neither failed nor unrunnable. Not marking the end node nor'
          ' its descendants as unrunnable due to the failures of %s.',
          q_node_id,
          start_node,
          ', '.join(failed_node_ids),
      )
      continue
    if q_node_id not in unrunnable:
      queue.extend(node_by_id[q_node_id].downstream_nodes)
      unrunnable.add(q_node_id)

  # Lazy nodes whose descendents are all unrunnable are also unrunnable, so we
  # need to add them here.
  # We go over the dictionary in reverse order so that lazy nodes that are
  # downstream of other lazy nodes are checked in (reverse) order.
  for node_id, node in reversed(node_by_id.items()):
    if (
        node.execution_options.strategy in _LAZY_TRIGGER_STRATEGIES
        and node.downstream_nodes
        and all(
            downstream in unrunnable for downstream in node.downstream_nodes
        )
    ):
      unrunnable.add(node_id)
  return unrunnable
