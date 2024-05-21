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
"""Pipeline-level operations."""

import collections
import contextlib
import copy
import dataclasses
import datetime
import functools
import itertools
import os
import random
import threading
import time
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from absl import logging
import attr
from tfx import types
from tfx.dsl.io import fileio
from tfx.dsl.io import filesystem
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core.task_schedulers import manual_task_scheduler
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable.mlmd import artifact_lib
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import io_utils
from tfx.utils import status as status_lib

from ml_metadata import errors as mlmd_errors
from ml_metadata.proto import metadata_store_pb2


# A coarse grained lock is used to ensure serialization of pipeline operations
# since there isn't a suitable MLMD transaction API.
_PIPELINE_OPS_LOCK = threading.RLock()

# Default polling interval to be used with `_wait_for_predicate` function when
# the predicate_fn is expected to perform in-memory operations (discounting
# cache misses).
_IN_MEMORY_PREDICATE_FN_DEFAULT_POLLING_INTERVAL_SECS = 1.0

# A special message indicating that a node is stopped by the command Update.
_STOPPED_BY_UPDATE = 'Stopped by Update command'


def _pipeline_op(lock: bool = True):
  """Decorator factory for pipeline ops."""

  def _decorator(fn):
    """Decorator for pipeline ops."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
      with contextlib.ExitStack() as stack:
        if lock:
          stack.enter_context(_PIPELINE_OPS_LOCK)

        health_status = env.get_env().health_status()
        if health_status.code != status_lib.Code.OK:
          raise status_lib.StatusNotOkError(
              code=health_status.code,
              message=(
                  'Operation cannot be completed because the Orchestrator is'
                  f' unhealthy. Error: {health_status.message}'
              ),
          )

        try:
          return fn(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
          logging.exception('Error raised by `%s`:', fn.__name__)
          if isinstance(e, status_lib.StatusNotOkError):
            raise
          raise status_lib.StatusNotOkError(
              code=status_lib.Code.UNKNOWN,
              message=f'`{fn.__name__}` error: {str(e)}',
          ) from e

    return _wrapper

  return _decorator


@_pipeline_op()
def initiate_pipeline_start(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    pipeline_run_metadata: Optional[Mapping[str, types.Property]] = None,
    partial_run_option: Optional[pipeline_pb2.PartialRun] = None,
) -> pstate.PipelineState:
  """Initiates a pipeline start operation.

  Upon success, MLMD is updated to signal that the pipeline must be started.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: IR of the pipeline to start.
    pipeline_run_metadata: Pipeline run metadata.
    partial_run_option: Options for partial pipeline run.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline start. With code
      `INVALILD_ARGUMENT` if it's a sync pipeline without `pipeline_run_id`
      provided.
  """
  logging.info(
      'Received request to start pipeline; pipeline uid: %s',
      task_lib.PipelineUid.from_pipeline(pipeline),
  )
  env.get_env().check_if_can_orchestrate(pipeline)
  pipeline = copy.deepcopy(pipeline)

  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC and not (
      pipeline.runtime_spec.pipeline_run_id.HasField('field_value')
      and pipeline.runtime_spec.pipeline_run_id.field_value.string_value
  ):
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message='Sync pipeline IR must specify pipeline_run_id.',
    )

  reused_pipeline_view = None
  if partial_run_option:
    if pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message='Partial pipeline run is not supported for async pipelines.',
      )
    snapshot_settings = partial_run_option.snapshot_settings
    which_strategy = snapshot_settings.WhichOneof('artifact_reuse_strategy')
    if which_strategy is None:
      logging.info(
          'No artifact_reuse_strategy specified for the partial pipeline run, '
          'defaulting to latest_pipeline_run_strategy.'
      )
      partial_run_utils.set_latest_pipeline_run_strategy(snapshot_settings)
    reused_pipeline_view = _load_reused_pipeline_view(
        mlmd_handle, pipeline, partial_run_option.snapshot_settings
    )
    # Mark nodes using partial pipeline run lib.
    # Nodes marked as SKIPPED (due to conditional) do not have an execution
    # registered in MLMD, so we skip their snapshotting step.
    try:
      pipeline = partial_run_utils.mark_pipeline(
          pipeline,
          from_nodes=partial_run_option.from_nodes,
          to_nodes=partial_run_option.to_nodes,
          skip_nodes=partial_run_option.skip_nodes,
          skip_snapshot_nodes=_get_previously_skipped_nodes(
              reused_pipeline_view
          ),
          snapshot_settings=partial_run_option.snapshot_settings,
      )
    except ValueError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT, message=str(e)
      )
    else:
      # Find all subpipelines in the parent pipeline, which we are caching.
      to_process = collections.deque([])
      for node in pipeline.nodes:
        # Only add to processing queue if it's a subpipeline that we are going
        # to cache. For subpipelines, the begin node's (nodes[0]) execution
        # options represent the subpipeline's execution options.
        if node.WhichOneof(
            'node'
        ) == 'sub_pipeline' and partial_run_utils.should_attempt_to_reuse_artifact(
            node.sub_pipeline.nodes[0].pipeline_node.execution_options
        ):
          to_process.append(node.sub_pipeline)
      cached_subpipelines = []
      while to_process:
        subpipeline = to_process.popleft()
        cached_subpipelines.append(subpipeline)
        to_process.extend(
            node.sub_pipeline
            for node in subpipeline.nodes
            if node.WhichOneof('node') == 'sub_pipeline'
        )
      logging.info(
          'Found subpipelines: %s',
          [s.pipeline_info.id for s in cached_subpipelines],
      )
      # Add a new pipeline run for every subpipeline we are going to cache in
      # the partial run.
      for subpipeline in cached_subpipelines:
        reused_subpipeline_view = _load_reused_pipeline_view(
            mlmd_handle, subpipeline, partial_run_option.snapshot_settings
        )
        # TODO: b/323912217 - Support putting multiple subpipeline executions
        # into MLMD to handle the ForEach case.
        with pstate.PipelineState.new(
            mlmd_handle,
            subpipeline,
            pipeline_run_metadata,
            reused_subpipeline_view,
        ) as subpipeline_state:
          # TODO: b/320535460 - The new pipeline run should not be stopped if
          # there are still nodes to run in it.
          logging.info('Subpipeline execution cached for partial run.')
          subpipeline_state.initiate_stop(
              status_lib.Status(
                  code=status_lib.Code.OK,
                  message='Subpipeline execution cached for partial run.',
              )
          )
  if pipeline.runtime_spec.HasField('snapshot_settings'):
    try:
      base_run_id = (
          reused_pipeline_view.pipeline_run_id if reused_pipeline_view else None
      )
      partial_run_utils.snapshot(mlmd_handle, pipeline, base_run_id)
    except ValueError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT, message=str(e)
      )
    except LookupError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION, message=str(e)
      )
  env.get_env().prepare_orchestrator_for_pipeline_run(pipeline)
  return pstate.PipelineState.new(
      mlmd_handle, pipeline, pipeline_run_metadata, reused_pipeline_view
  )


@_pipeline_op(lock=False)
def stop_pipelines(
    mlmd_handle: metadata.Metadata,
    pipeline_uids: List[task_lib.PipelineUid],
    return_immediately: bool = False,
    timeout_secs: Optional[float] = None,
    ignore_non_existent_or_inactive: Optional[bool] = False,
) -> None:
  """Stops multiple pipelines.

  Initiates pipeline stop operations and waits for the pipeline executions to be
  gracefully stopped in the orchestration loop.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_uids: UIDs of the pipeline to be stopped.
    return_immediately: If true, returns immediately to skip waiting for all
      pipelines to be inactive. If false, waits for all the pipelines to
      completely stop before returning.
    timeout_secs: Amount of time in seconds total to wait for all pipelines to
      stop. If `None`, waits indefinitely.
    ignore_non_existent_or_inactive: If a pipeline is not found or inactive,
      skips it. This is useful if pipeline uids contain nested pipelines.
      Stopping outer pipeline automatically stops inner pipelines, hence we may
      need to skip inner pipelines here.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate pipeline stop.
  """
  pipeline_ids_str = ', '.join([x.pipeline_id for x in pipeline_uids])
  pipeline_states = []
  logging.info(
      'Received request to stop pipelines; pipeline ids: %s', pipeline_ids_str
  )
  with _PIPELINE_OPS_LOCK:
    for pipeline_uid in pipeline_uids:
      try:
        with pstate.PipelineState.load(
            mlmd_handle, pipeline_uid
        ) as pipeline_state:
          env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
          pipeline_state.initiate_stop(
              status_lib.Status(
                  code=status_lib.Code.CANCELLED,
                  message='Cancellation requested by client.',
              )
          )
          pipeline_states.append(pipeline_state)
      except status_lib.StatusNotOkError as e:
        if (
            e.code == status_lib.Code.NOT_FOUND
            and ignore_non_existent_or_inactive
        ):
          logging.info(
              'Ignored non-existent or inactive pipeline %s.', pipeline_uid
          )
          continue
        raise e

  if return_immediately:
    logging.info(
        'Skipping wait for all pipelines to be inactive; pipeline ids: %s.',
        pipeline_ids_str,
    )
    return

  logging.info(
      'Waiting for pipelines to be stopped; pipeline ids: %s', pipeline_ids_str
  )

  def _are_pipelines_inactivated() -> bool:
    for pipeline_state in pipeline_states:
      with pipeline_state:
        if pipeline_state.is_active():
          return False
    return True

  _wait_for_predicate(
      _are_pipelines_inactivated,
      'inactivation of pipelines',
      _IN_MEMORY_PREDICATE_FN_DEFAULT_POLLING_INTERVAL_SECS,
      timeout_secs,
  )
  logging.info(
      'Done waiting for pipelines to be stopped; pipeline ids: %s',
      pipeline_ids_str,
  )


@_pipeline_op(lock=False)
def stop_pipeline(
    mlmd_handle: metadata.Metadata,
    pipeline_uid: task_lib.PipelineUid,
    return_immediately: bool = False,
    timeout_secs: Optional[float] = None,
) -> None:
  """Stops a single pipeline. Convenience wrapper around stop_pipelines."""
  return stop_pipelines(
      mlmd_handle=mlmd_handle,
      pipeline_uids=[pipeline_uid],
      timeout_secs=timeout_secs,
      return_immediately=return_immediately,
  )


# TODO(b/285976181): Support retrying individual pipelines nodes from a stopped
# pipeline.
@_pipeline_op()
def initiate_node_start(
    mlmd_handle: metadata.Metadata, node_uid: task_lib.NodeUid
) -> pstate.PipelineState:
  """Initiates a node start operation for a pipeline node.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the node to be started.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate node start operation.
  """
  logging.info('Received request to start node; node uid: %s', node_uid)
  with pstate.PipelineState.load(
      mlmd_handle, node_uid.pipeline_uid
  ) as pipeline_state:
    env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
    with pipeline_state.node_state_update_context(node_uid) as node_state:
      if node_state.is_startable():
        node_state.update(pstate.NodeState.STARTED)
  return pipeline_state


@_pipeline_op()
def initiate_node_backfill(
    mlmd_handle: metadata.Metadata, node_uid: task_lib.NodeUid
) -> None:
  """Initiates a node backfill operation for a pipeline node.

  Only works on ASYNC pipelines. Doesn't work on nodes within subpipelines.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the node to be backfilled.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to initiate node backfill operation.
  """
  logging.info('Received request to backfill node; node uid: %s', node_uid)
  with pstate.PipelineState.load(
      mlmd_handle, node_uid.pipeline_uid
  ) as pipeline_state:
    env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
    if pipeline_state.pipeline.execution_mode != pipeline_pb2.Pipeline.ASYNC:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(
              'Can only backfill nodes in an ASYNC pipeline, but pipeline '
              f'{node_uid.pipeline_uid.pipeline_id} is not ASYNC'
          ),
      )

    with pipeline_state.node_state_update_context(node_uid) as node_state:
      if node_state.backfill_token:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INVALID_ARGUMENT,
            message=(
                f'Node {node_uid} is already in backfill mode with token '
                f'{node_state.backfill_token}. If you want to abort the '
                'backfill and start a new one, stop the node first.'
            ),
        )

      if node_state.is_backfillable():
        # Generate a unique backfill token for this request.
        backfill_token = 'backfill-%s-%06s' % (
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            random.randint(0, 999999),
        )
        node_state.update(
            pstate.NodeState.STARTED, backfill_token=backfill_token
        )
      else:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INVALID_ARGUMENT,
            message=(
                'Can only backfill nodes in a stopped or failed state, '
                f'but node {node_uid} was in state {node_state.state}. '
                'Try stopping the node first.'
            ),
        )


def _check_nodes_exist(
    node_uids: Sequence[task_lib.NodeUid],
    pipeline: pipeline_pb2.Pipeline,
    op_name: str,
) -> None:
  """Raises an error if node_uid does not exist in the pipeline."""
  node_id_set = set(n.node_id for n in node_uids)
  nodes = pstate.get_all_nodes(pipeline)
  filtered_nodes = [n for n in nodes if n.node_info.id in node_id_set]
  if len(filtered_nodes) != len(node_id_set):
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message=(
            f'`f{op_name}` operation failed, cannot find node(s) '
            f'{", ".join(node_id_set)} in the pipeline IR.'
        ),
    )


@_pipeline_op(lock=False)
def stop_node(
    mlmd_handle: metadata.Metadata,
    node_uid: task_lib.NodeUid,
    timeout_secs: Optional[float] = None,
) -> None:
  """Stops a node.

  Initiates a node stop operation and waits for the node execution to become
  inactive.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the node to be stopped.
    timeout_secs: Amount of time in seconds to wait for node to stop. If `None`,
      waits indefinitely.

  Raises:
    status_lib.StatusNotOkError: Failure to stop the node.
  """
  logging.info('Received request to stop node; node uid: %s', node_uid)
  with _PIPELINE_OPS_LOCK:
    with pstate.PipelineState.load(
        mlmd_handle, node_uid.pipeline_uid
    ) as pipeline_state:
      env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
      _check_nodes_exist([node_uid], pipeline_state.pipeline, 'stop_node')
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.is_stoppable():
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(
                  code=status_lib.Code.CANCELLED,
                  message='Cancellation requested by client.',
              ),
          )

  # Wait until the node is stopped or time out.
  _wait_for_node_inactivation(
      pipeline_state, node_uid, timeout_secs=timeout_secs
  )


@_pipeline_op()
def skip_nodes(
    mlmd_handle: metadata.Metadata, node_uids: Sequence[task_lib.NodeUid]
) -> None:
  """Marks node executions to be skipped."""
  # All node_uids must have the same pipeline_uid.
  pipeline_uids_set = set(n.pipeline_uid for n in node_uids)
  if len(pipeline_uids_set) != 1:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message='Can skip nodes of a single pipeline at once.',
    )
  pipeline_uid = pipeline_uids_set.pop()
  with pstate.PipelineState.load(mlmd_handle, pipeline_uid) as pipeline_state:
    env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
    _check_nodes_exist(node_uids, pipeline_state.pipeline, 'skip_nodes')
    for node_uid in node_uids:
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.state == pstate.NodeState.SKIPPED:
          continue
        elif node_state.is_programmatically_skippable():
          node_state.update(
              pstate.NodeState.SKIPPED,
              status_lib.Status(
                  code=status_lib.Code.OK,
                  message='Node skipped by client request.',
              ),
          )
        else:
          raise status_lib.StatusNotOkError(
              code=status_lib.Code.FAILED_PRECONDITION,
              message=(
                  f'Node in state {node_state.state} is not programmatically'
                  ' skippable.'
              ),
          )


@_pipeline_op()
def resume_manual_node(
    mlmd_handle: metadata.Metadata, node_uid: task_lib.NodeUid
) -> None:
  """Resumes a manual node.

  Args:
    mlmd_handle: A handle to the MLMD db.
    node_uid: Uid of the manual node to be resumed.

  Raises:
    status_lib.StatusNotOkError: Failure to resume a manual node.
  """
  logging.info('Received request to resume manual node; node uid: %s', node_uid)
  with pstate.PipelineState.load(
      mlmd_handle, node_uid.pipeline_uid
  ) as pipeline_state:
    env.get_env().check_if_can_orchestrate(pipeline_state.pipeline)
    nodes = pstate.get_all_nodes(pipeline_state.pipeline)
    filtered_nodes = [n for n in nodes if n.node_info.id == node_uid.node_id]
    if len(filtered_nodes) != 1:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.NOT_FOUND,
          message=f'Unable to find manual node to resume: {node_uid}',
      )
    node = filtered_nodes[0]
    node_type = node.node_info.type.name
    if node_type != constants.MANUAL_NODE_TYPE:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT,
          message=(
              'Unable to resume a non-manual node. '
              f'Got non-manual node id: {node_uid}'
          ),
      )

  executions = task_gen_utils.get_executions(mlmd_handle, node)
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=f'Unable to find active manual node to resume: {node_uid}',
    )
  if len(active_executions) > 1:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(
            f'Unexpected multiple active executions for manual node: {node_uid}'
        ),
    )
  with mlmd_state.mlmd_execution_atomic_op(
      mlmd_handle=mlmd_handle, execution_id=active_executions[0].id
  ) as execution:
    completed_state = manual_task_scheduler.ManualNodeState(
        state=manual_task_scheduler.ManualNodeState.COMPLETED
    )
    completed_state.set_mlmd_value(
        execution.custom_properties.get_or_create(
            manual_task_scheduler.NODE_STATE_PROPERTY_KEY
        )
    )


@_pipeline_op()
def _initiate_pipeline_update(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    update_options: pipeline_pb2.UpdateOptions,
) -> pstate.PipelineState:
  """Initiates pipeline update."""
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  with pstate.PipelineState.load(mlmd_handle, pipeline_uid) as pipeline_state:
    pipeline_state.initiate_update(pipeline, update_options)
  return pipeline_state


@_pipeline_op()
def delete_pipeline_run(
    mlmd_handle: metadata.Metadata, pipeline_id: str, pipeline_run_id: str
) -> None:
  """Deletes a pipeline run.

  Mark the pipeline run execution custom_priority['deleted'] to true and
  pipeline run output artifacts as DELETED.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_id: id of the pipeline which has the pipeline run.
    pipeline_run_id: id of the pipeline run will be deleted.

  Raises:
     status_lib.StatusNotOkError: Failure to delete a pipeline run.
  """
  try:
    pipeline_view = pstate.PipelineView.load(
        mlmd_handle, pipeline_id, pipeline_run_id
    )
    # No orchestration is required for delete, so we don't have to check
    # whether we can orchestrate this pipeline or not.
    if (
        pipeline_view.pipeline_execution_mode
        == pipeline_pb2.Pipeline.ExecutionMode.ASYNC
    ):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message='delete pipeline run does not support ASYNC pipeline',
      )
    if (
        pipeline_view.execution.last_known_state
        == mlmd_state.metadata_store_pb2.Execution.State.RUNNING
    ):
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              "Tflex doesn't allow deleting the active running pipeline run,"
              ' please stop the pipeline run first.'
          ),
      )
    # mark executions as deleted using atomic op to avoid race condition.
    with mlmd_state.mlmd_execution_atomic_op(
        mlmd_handle=mlmd_handle,
        execution_id=pipeline_view.execution.id,
    ) as execution:
      if not execution:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.NOT_FOUND,
            message=(
                'Execution with given execution_id not found: '
                f'{pipeline_view.execution.id}'
            ),
        )
      execution.custom_properties['deleted'].CopyFrom(
          mlmd_state.metadata_store_pb2.Value(bool_value=True)
      )

      # TODO(fangyuancai):consider using atomic operation when modify artifacts.
      artifacts = []
      artifacts_dict = pstate.get_all_node_artifacts(
          pipeline_view.pipeline, mlmd_handle
      )
      for _, node_artifacts in artifacts_dict.items():
        for _, execution_artifacts in node_artifacts.items():
          for _, artifact_list in execution_artifacts.items():
            artifacts.extend(artifact_list)
      for artifact in artifacts:
        artifact.state = mlmd_state.metadata_store_pb2.Artifact.State.DELETED
        try:
          io_utils.delete_dir(artifact.uri)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.warning(
              "The artifact's uri is not a directory. We will mark it as"
              ' DELETED in MLMD but keep the path'
          )

    mlmd_handle.store.put_artifacts(artifacts)
  except LookupError as e:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND, message=str(e)
    )


@_pipeline_op(lock=False)
def update_pipeline(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    update_options: pipeline_pb2.UpdateOptions,
    timeout_secs: Optional[float] = None,
) -> None:
  """Updates an active pipeline with a new pipeline IR.

  Initiates a pipeline update operation and waits for it to finish.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: New pipeline IR to be applied.
    update_options: Selection of active nodes to be reloaded upon update.
    timeout_secs: Timeout in seconds to wait for the update to finish. If
      `None`, waits indefinitely.

  Raises:
    status_lib.StatusNotOkError: Failure to update the pipeline.
  """
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  logging.info(
      'Received request to update pipeline; pipeline uid: %s', pipeline_uid
  )
  env.get_env().check_if_can_orchestrate(pipeline)
  pipeline_state = _initiate_pipeline_update(
      mlmd_handle, pipeline, update_options
  )

  def _is_update_applied() -> bool:
    with pipeline_state:
      if pipeline_state.is_active():
        return not pipeline_state.is_update_initiated()
      # If the pipeline is no longer active, whether or not the update is
      # applied is irrelevant.
      return True

  logging.info('Waiting for pipeline update; pipeline uid: %s', pipeline_uid)
  _wait_for_predicate(
      _is_update_applied,
      'pipeline update',
      _IN_MEMORY_PREDICATE_FN_DEFAULT_POLLING_INTERVAL_SECS,
      timeout_secs,
  )
  logging.info(
      'Done waiting for pipeline update; pipeline uid: %s', pipeline_uid
  )


def _wait_for_node_inactivation(
    pipeline_state: pstate.PipelineState,
    node_uid: task_lib.NodeUid,
    timeout_secs: Optional[float],
) -> None:
  """Waits for the given node to become inactive.

  Args:
    pipeline_state: Pipeline state.
    node_uid: Uid of the node whose inactivation is awaited.
    timeout_secs: Amount of time in seconds to wait. If `None`, waits
      indefinitely.

  Raises:
    StatusNotOkError: With error code `DEADLINE_EXCEEDED` if node is not
      inactive after waiting approx. `timeout_secs`.
  """

  def _is_inactivated() -> bool:
    with pipeline_state:
      node_state = pipeline_state.get_node_state(node_uid)
      return node_state.state in (
          pstate.NodeState.COMPLETE,
          pstate.NodeState.FAILED,
          pstate.NodeState.SKIPPED,
          pstate.NodeState.STOPPED,
      )

  _wait_for_predicate(
      _is_inactivated,
      'node inactivation',
      _IN_MEMORY_PREDICATE_FN_DEFAULT_POLLING_INTERVAL_SECS,
      timeout_secs,
  )


def _get_previously_skipped_nodes(
    reused_pipeline_view: Optional[pstate.PipelineView],
) -> List[str]:
  """Returns id of nodes skipped in previous pipeline run due to conditional."""
  reused_pipeline_node_states = (
      reused_pipeline_view.get_node_states_dict()
      if reused_pipeline_view
      else dict()
  )
  reused_pipeline_previous_node_states = (
      reused_pipeline_view.get_previous_node_states_dict()
      if reused_pipeline_view
      else dict()
  )
  skipped_nodes = []
  for node_id, node_state in itertools.chain(
      reused_pipeline_node_states.items(),
      reused_pipeline_previous_node_states.items(),
  ):
    if node_state.state == pstate.NodeState.SKIPPED:
      skipped_nodes.append(node_id)
  return skipped_nodes


def _load_reused_pipeline_view(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    snapshot_settings: pipeline_pb2.SnapshotSettings,
) -> Optional[pstate.PipelineView]:
  """Loads pipeline view of the pipeline reused for partial pipeline run."""
  base_run_id = None
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  if snapshot_settings.HasField('base_pipeline_run_strategy'):
    base_run_id = snapshot_settings.base_pipeline_run_strategy.base_run_id
  try:
    reused_pipeline_view = pstate.PipelineView.load(
        mlmd_handle=mlmd_handle,
        pipeline_id=pipeline_uid.pipeline_id,
        pipeline_run_id=base_run_id,
        # If current pipeline run is allowed and base_run_id is not specified,
        # reuse the most recent completed run.
        non_active_only=env.get_env().concurrent_pipeline_runs_enabled(
            pipeline
        ),
    )
  except status_lib.StatusNotOkError as e:
    if e.code == status_lib.Code.NOT_FOUND:
      # A previous pipeline run is not strictly required, since users are
      # allowed to start a partial run without reusing any nodes. Returns None
      # to delay the error handling to caller function.
      logging.info(e.message)
      return None
    else:
      raise

  if reused_pipeline_view.pipeline.execution_mode != pipeline_pb2.Pipeline.SYNC:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(
            'Only SYNC pipeline execution modes supported; previous pipeline '
            'run has execution mode: '
            f'{reused_pipeline_view.pipeline.execution_mode}'
        ),
    )

  if execution_lib.is_execution_active(reused_pipeline_view.execution):
    if base_run_id and env.get_env().concurrent_pipeline_runs_enabled(pipeline):
      # TODO(b/330376413): Ideally we should not allow an active run to be
      # reused, otherwise the new partial run may end up in an invalid state due
      # to race condition. But there are users who already depend on this buggy
      # behavior, so we keep it as is for now.
      logging.warning(
          'The base pipeline run %s is still active. The new partial run'
          ' may end up in an invalid state due to race condition.',
          base_run_id,
      )
    else:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              'The base pipeline run'
              f' {reused_pipeline_view.pipeline_run_id} is still active.'
          ),
      )

  return reused_pipeline_view


@_pipeline_op()
def resume_pipeline(
    mlmd_handle: metadata.Metadata,
    pipeline: pipeline_pb2.Pipeline,
    run_id: Optional[str] = None,
) -> pstate.PipelineState:
  """Resumes a pipeline run from previously failed nodes.

  Upon success, MLMD is updated to signal that the pipeline must be started.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: IR of the pipeline to resume.
    run_id: the run_id of the pipeline run to resume.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to resume pipeline. With code
      `ALREADY_EXISTS` if a pipeline is already running. With code
      `status_lib.Code.FAILED_PRECONDITION` if a previous pipeline run
      is not found for resuming. With code 'INVALID_ARGUMENT' if concurrent
      pipeline runs are enabled but pipeline run id is missing.
  """
  logging.info(
      'Received request to resume pipeline; pipeline uid: %s',
      task_lib.PipelineUid.from_pipeline(pipeline),
  )
  if pipeline.execution_mode != pipeline_pb2.Pipeline.SYNC:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(
            'Only SYNC pipeline execution modes supported; '
            f'found pipeline with execution mode: {pipeline.execution_mode}'
        ),
    )

  if env.get_env().concurrent_pipeline_runs_enabled(pipeline) and not run_id:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message=(
            'Pipeline Run ID of the old pipeline to resume must be '
            'provided when concurrent pipeline runs are enabled.'
        ),
    )

  if run_id:
    snapshot_settings = pipeline_pb2.SnapshotSettings()
    partial_run_utils.set_base_pipeline_run_strategy(
        snapshot_settings, run_id
    )
  else:
    snapshot_settings = partial_run_utils.latest_pipeline_snapshot_settings()

  latest_pipeline_view = _load_reused_pipeline_view(
      mlmd_handle, pipeline, snapshot_settings
  )
  if not latest_pipeline_view:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message='Pipeline failed to resume. No previous pipeline run found.',
    )
  # TODO(b/200206549): Remove once testing is complete
  # Get succeeded nodes in latest pipeline run.
  previously_succeeded_nodes = []
  for node, node_state in latest_pipeline_view.get_node_states_dict().items():
    if node_state.is_success():
      previously_succeeded_nodes.append(node)
  pipeline_nodes = [
      node.node_info.id for node in pstate.get_all_nodes(pipeline)
  ]

  # Mark nodes using partial pipeline run lib.
  # Nodes marked as SKIPPED (due to conditional) do not have an execution
  # registered in MLMD, so we skip their snapshotting step.
  try:
    pipeline = partial_run_utils.mark_pipeline(
        pipeline,
        from_nodes=pipeline_nodes,
        to_nodes=pipeline_nodes,
        skip_nodes=previously_succeeded_nodes,
        skip_snapshot_nodes=_get_previously_skipped_nodes(
            latest_pipeline_view
        ),
        snapshot_settings=snapshot_settings,
    )
  except ValueError as e:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT, message=str(e)
    )
  if pipeline.runtime_spec.HasField('snapshot_settings'):
    try:
      partial_run_utils.snapshot(
          mlmd_handle, pipeline, latest_pipeline_view.pipeline_run_id
      )
    except ValueError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.INVALID_ARGUMENT, message=str(e)
      )
    except LookupError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION, message=str(e)
      )
  env.get_env().prepare_orchestrator_for_pipeline_run(pipeline)
  return pstate.PipelineState.new(
      mlmd_handle, pipeline, reused_pipeline_view=latest_pipeline_view
  )


def _recursively_revive_pipelines(
    mlmd_handle: metadata.Metadata,
    pipeline_state: pstate.PipelineState,
) -> pstate.PipelineState:
  """Recursively revives all pipelines, resuing executions if present."""
  with pipeline_state:
    nodes = pstate.get_all_nodes(pipeline_state.pipeline)
    node_by_name = {node.node_info.id: node for node in nodes}
    # TODO(b/272015049): Add support for manager start nodes.
    nodes_to_start = [
        node_uid
        for node_uid, state in pipeline_state.get_node_states_dict().items()
        if state.is_startable()
    ]

    logging.info(
        'The following nodes will be attempted to be started: %s',
        [node.node_id for node in nodes_to_start],
    )
    for node_uid in nodes_to_start:
      new_node_state = pstate.NodeState.STARTED
      node = node_by_name[node_uid.node_id]
      # Subpipelines are represented in their parent pipeline as node,
      # so to revive the full pipeline in place we need to peer into the
      # subpipeline.
      if isinstance(node, node_proto_view.ComposablePipelineProtoView):
        subpipeline_base_run_id = (
            node.raw_proto().runtime_spec.pipeline_run_id.field_value.string_value
        )
        logging.info(
            '%s is a subpipeline, run_id: %s',
            node.node_info.id,
            subpipeline_base_run_id,
        )

        # Subpipeline run id's are structured like:
        # ${SUBPIPELINE_ID}_${PARENT_PIPELINE_ID}_${SUBPIPELINE_EXECUTION_ID}
        # So we need to determine the execution id for the pipeline so it can
        # be revived. If there's no execution found then assume it hasn't been
        # run so it can be marked as STARTED.
        executions = task_gen_utils.get_executions(mlmd_handle, node)
        latest_execution_set = task_gen_utils.get_latest_executions_set(
            executions
        )
        logging.info(
            'Executions for subpipeline %s: %s',
            node.node_info.id,
            [
                f'{e.id}: state:'
                f' {metadata_store_pb2.Execution.State.Name(e.last_known_state)}'
                for e in latest_execution_set
            ],
        )
        if not latest_execution_set:
          logging.info(
              'No executions found for subpipeline %s, marking as STARTED.',
              node.node_info.id,
          )
          new_node_state = pstate.NodeState.STARTED
        elif all(
            execution_lib.is_execution_successful(execution)
            for execution in latest_execution_set
        ):
          logging.info(
              'All executions in subpipeline %s were SUCCESSFUL, will mark as'
              ' COMPLETE.',
              node.node_info.id,
          )
          new_node_state = pstate.NodeState.COMPLETE
        else:
          # Mark all subpipeline executions as NEW, and the node state as
          # RUNNING.
          new_node_state = pstate.NodeState.RUNNING
          non_successful_executions = [
              e
              for e in latest_execution_set
              if not execution_lib.is_execution_successful(e)
          ]
          for execution in non_successful_executions:
            # TODO: b/324962451 - Consolidate all subpipeline run naming into a
            # utility function.
            new_run_id = f'{subpipeline_base_run_id}_{execution.id}'
            # Potentially, a subpipeline execution can be CANCELLED but have
            # never started, for instance if it's in the second iteration of
            # ForEach. In this case we *do not* want to revive recursively, as
            # there is no pipeline run started.
            try:
              subpipeline_state = pstate.PipelineState.load_run(
                  mlmd_handle, pipeline_id=node.node_info.id, run_id=new_run_id
              )
            except status_lib.StatusNotOkError:
              logging.info(
                  'Failed to load run %s of pipeline %s. Assuming there is no'
                  ' existing run.',
                  new_run_id,
                  node.node_info.id,
              )
            else:
              _recursively_revive_pipelines(
                  mlmd_handle,
                  subpipeline_state,
              )
            # Mark the execution as NEW and the node state as RUNNING so we can
            # re-use the existing execution during task generation.
            with mlmd_state.mlmd_execution_atomic_op(
                mlmd_handle, execution.id
            ) as execution:
              logging.info(
                  'Execution for subpipeline %s: %s. Changing from state %s'
                  ' to %s.',
                  node.node_info.id,
                  execution.id,
                  metadata_store_pb2.Execution.State.Name(
                      execution.last_known_state
                  ),
                  metadata_store_pb2.Execution.State.Name(
                      metadata_store_pb2.Execution.State.NEW
                  ),
              )
              execution.last_known_state = (
                  metadata_store_pb2.Execution.State.NEW
              )
              if execution.custom_properties.get(
                  constants.EXECUTION_ERROR_CODE_KEY
              ):
                del execution.custom_properties[
                    constants.EXECUTION_ERROR_CODE_KEY
                ]
              if execution.custom_properties.get(
                  constants.EXECUTION_ERROR_MSG_KEY
              ):
                del execution.custom_properties[
                    constants.EXECUTION_ERROR_MSG_KEY
                ]
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        node_state.update(new_node_state)

    pipeline_state.initiate_resume()
    new_pipeline_state = metadata_store_pb2.Execution.State.NEW
    pipeline_state.set_pipeline_execution_state(new_pipeline_state)
    return pipeline_state


@_pipeline_op()
def revive_pipeline_run(
    mlmd_handle: metadata.Metadata,
    pipeline_id: str,
    pipeline_run_id: str,
    pipeline_to_update_with: Optional[pipeline_pb2.Pipeline] = None,
) -> pstate.PipelineState:
  """Revives a pipeline run from previously failed nodes.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_id: The id (name) of the pipeline to resume.
    pipeline_run_id: the run_id of the pipeline run to resume.
    pipeline_to_update_with: Optionally an IR to update to for the revived run.

  Returns:
    The `PipelineState` object upon success.

  Raises:
    status_lib.StatusNotOkError: Failure to resume pipeline. With code
      `ALREADY_EXISTS` if a pipeline is already running. With code
      `status_lib.Code.FAILED_PRECONDITION` if a previous pipeline run
      is not found for resuming. With code 'INVALID_ARGUMENT' if trying to
      revive a pipeline run while there's another active run and concurrent runs
      are not enabled.
  """
  logging.info(
      'Received request to revive run %s of pipeline %s',
      pipeline_run_id,
      pipeline_id,
  )

  with pstate.PipelineState.load_run(
      mlmd_handle, pipeline_id=pipeline_id, run_id=pipeline_run_id
  ) as pipeline_state:
    pipeline = pipeline_state.pipeline
    if pipeline.execution_mode != pipeline_pb2.Pipeline.SYNC:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(
              'Only SYNC pipeline execution modes supported; '
              f'but pipeline had execution mode: {pipeline.execution_mode}'
          ),
      )
    if pipeline_state.is_active():
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS,
          message='Cannot revive a live pipeline run.',
      )

    # Since the pipeline is not active we can apply the update right away.
    if pipeline_to_update_with is not None:
      logging.info('Trying to update during revive')
      pipeline_state.initiate_update(
          pipeline_to_update_with, pipeline_pb2.UpdateOptions()
      )
      logging.info('Initiated update')
      pipeline_state.apply_pipeline_update()
      logging.info('Applied update')

  revived_pipeline_state = _recursively_revive_pipelines(
      mlmd_handle, pipeline_state
  )
  return revived_pipeline_state


def _wait_for_predicate(
    predicate_fn: Callable[[], bool],
    waiting_for_desc: str,
    polling_interval_secs: float,
    timeout_secs: Optional[float],
) -> None:
  """Waits for `predicate_fn` to return `True` or until timeout seconds elapse."""
  if timeout_secs is None:
    while not predicate_fn():
      logging.info(
          'Sleeping %f sec(s) waiting for predicate: %s',
          polling_interval_secs,
          waiting_for_desc,
      )
      time.sleep(polling_interval_secs)
    return
  polling_interval_secs = min(polling_interval_secs, timeout_secs / 4)
  end_time = time.time() + timeout_secs
  while end_time - time.time() > 0:
    if predicate_fn():
      return
    sleep_secs = max(0, min(polling_interval_secs, end_time - time.time()))
    logging.info(
        'Sleeping %f sec(s) waiting for predicate: %s',
        sleep_secs,
        waiting_for_desc,
    )
    time.sleep(sleep_secs)
  raise status_lib.StatusNotOkError(
      code=status_lib.Code.DEADLINE_EXCEEDED,
      message=(
          f'Timed out ({timeout_secs} secs) waiting for {waiting_for_desc}.'
      ),
  )


def filter_by_pipeline_uid(
    pipeline_uid: task_lib.PipelineUid,
) -> Callable[[pstate.PipelineState], bool]:
  """Returns filter_fn for orchestrate for the given pipeline_uid."""
  return lambda p: p.pipeline_uid == pipeline_uid


@_pipeline_op()
def orchestrate(
    mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
    task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    filter_fn: Optional[Callable[[pstate.PipelineState], bool]] = None,
) -> bool:
  """Performs a single iteration of the orchestration loop.

  Embodies the core functionality of the main orchestration loop that scans MLMD
  pipeline execution states, generates and enqueues the tasks to be performed.

  Args:
    mlmd_connection_manager: A `MLMDConnectionManager` instance to manager
      multiple mlmd connections.
    task_queue: A `TaskQueue` instance into which any tasks will be enqueued.
    service_job_manager: A `ServiceJobManager` instance for handling service
      jobs.
    filter_fn: Callable to filter pipelines to be orchestrated. Only active
      pipeline runs for which the filter_fn returns True will be orchestrated.
      If not provided, all active pipeline runs will be orchestrated.

  Returns:
    Whether there are any active pipelines to run.

  Raises:
    status_lib.StatusNotOkError: If error generating tasks.
  """
  if filter_fn is None:
    filter_fn = lambda _: True

  all_pipeline_states = pstate.PipelineState.load_all_active_and_owned(
      mlmd_connection_manager.primary_mlmd_handle
  )
  pipeline_states = [s for s in all_pipeline_states if filter_fn(s)]
  if not pipeline_states:
    logging.info('No active pipelines to run.')
    return False

  active_pipeline_states = []
  stop_initiated_pipeline_states = []
  update_initiated_pipeline_states = []
  for pipeline_state in pipeline_states:
    with pipeline_state:
      if pipeline_state.is_stop_initiated():
        stop_initiated_pipeline_states.append(pipeline_state)
      elif pipeline_state.is_update_initiated():
        update_initiated_pipeline_states.append(pipeline_state)
      elif pipeline_state.is_active():
        active_pipeline_states.append(pipeline_state)
      else:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.INTERNAL,
            message=(
                f'Found pipeline (uid: {pipeline_state.pipeline_uid}) '
                'which is neither active nor stop-initiated.'
            ),
        )

  for pipeline_state in stop_initiated_pipeline_states:
    logging.info(
        'Orchestrating stop-initiated pipeline: %s', pipeline_state.pipeline_uid
    )
    try:
      _orchestrate_stop_initiated_pipeline(
          mlmd_connection_manager,
          task_queue,
          service_job_manager,
          pipeline_state,
      )
    except Exception:  # pylint: disable=broad-except
      # If orchestrating a stop-initiated pipeline raises an exception, we log
      # the exception but do not re-raise since we do not want to crash the
      # orchestrator. If this issue persists across iterations of the
      # orchestration loop, the expectation is that user configured alerting
      # config will eventually fire alerts.
      logging.exception(
          'Exception raised while orchestrating stop-initiated pipeline %s',
          pipeline_state.pipeline_uid,
      )

  for pipeline_state in update_initiated_pipeline_states:
    logging.info(
        'Orchestrating update-initiated pipeline: %s',
        pipeline_state.pipeline_uid,
    )
    try:
      _orchestrate_update_initiated_pipeline(
          mlmd_connection_manager.primary_mlmd_handle,
          task_queue,
          service_job_manager,
          pipeline_state,
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          'Exception raised while orchestrating update-initiated pipeline %s',
          pipeline_state.pipeline_uid,
      )
      logging.info(
          'Attempting to initiate termination of update-initiated pipeline %s',
          pipeline_state.pipeline_uid,
      )
      try:
        with pipeline_state:
          pipeline_state.initiate_stop(
              status_lib.Status(
                  code=status_lib.Code.INTERNAL,
                  message=(
                      f'Error orchestrating update-initiated pipeline: {str(e)}'
                  ),
              )
          )
      except Exception:  # pylint: disable=broad-except
        # If stop initiation also raised an exception , we log the exception but
        # do not re-raise since we do not want to crash the orchestrator. If
        # this issue persists across iterations of the orchestration loop, the
        # expectation is that user configured alerting config will eventually
        # fire alerts.
        logging.exception(
            (
                'Error while attempting to terminate update-initiated pipeline'
                ' %s due to internal error'
            ),
            pipeline_state.pipeline_uid,
        )

  for pipeline_state in active_pipeline_states:
    logging.info('Orchestrating pipeline: %s', pipeline_state.pipeline_uid)
    try:
      _orchestrate_active_pipeline(
          mlmd_connection_manager,
          task_queue,
          service_job_manager,
          pipeline_state,
      )
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          'Exception raised while orchestrating active pipeline %s',
          pipeline_state.pipeline_uid,
      )
      logging.info(
          'Attempting to initiate termination of active pipeline %s',
          pipeline_state.pipeline_uid,
      )
      try:
        with pipeline_state:
          pipeline_state.initiate_stop(
              status_lib.Status(
                  code=status_lib.Code.INTERNAL,
                  message=f'Error orchestrating active pipeline: {str(e)}',
              )
          )
      except Exception:  # pylint: disable=broad-except
        # If stop initiation also raised an exception , we log the exception but
        # do not re-raise since we do not want to crash the orchestrator. If
        # this issue persists across iterations of the orchestration loop, the
        # expectation is that user configured alerting config will eventually
        # fire alerts.
        logging.exception(
            (
                'Error while attempting to terminate active pipeline %s due to'
                ' internal error'
            ),
            pipeline_state.pipeline_uid,
        )

  return True


def _cancel_node(
    mlmd_handle: metadata.Metadata,
    task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState,
    node: node_proto_view.NodeProtoView,
) -> bool:
  """Returns `True` if node cancelled successfully or no cancellation needed."""
  if service_job_manager.is_pure_service_node(
      pipeline_state, node.node_info.id
  ):
    node_uid = task_lib.NodeUid.from_node(pipeline_state.pipeline, node)
    logging.info('Stopping services for node: %s', node_uid)
    if service_job_manager.stop_node_services(
        pipeline_state, node.node_info.id
    ):
      logging.info(
          'Canceling active executions for pure service node: %s', node_uid
      )
      active_executions = task_gen_utils.get_executions(
          mlmd_handle,
          node,
          additional_filters=['last_known_state IN (NEW, RUNNING)'],
      )
      _cancel_executions(active_executions, mlmd_handle, node_uid)
      return True
    else:
      return False

  if _maybe_enqueue_cancellation_task(
      mlmd_handle, pipeline_state, node, task_queue
  ):
    return False

  if service_job_manager.is_mixed_service_node(
      pipeline_state, node.node_info.id
  ):
    return service_job_manager.stop_node_services(
        pipeline_state, node.node_info.id
    )

  return True


def _cancel_executions(
    executions: List[metadata_store_pb2.Execution],
    mlmd_handle: metadata.Metadata,
    node_uid: task_lib.NodeUid,
) -> None:
  """Cancels the given executions for the given node."""
  for execution in executions:
    previous_state = execution.last_known_state
    with mlmd_state.mlmd_execution_atomic_op(
        mlmd_handle=mlmd_handle,
        execution_id=execution.id,
        on_commit=event_observer.make_notify_execution_state_change_fn(
            node_uid
        ),
    ) as e:
      e.last_known_state = metadata_store_pb2.Execution.CANCELED
    if previous_state == metadata_store_pb2.Execution.RUNNING:
      pending_output_artifacts = execution_lib.get_pending_output_artifacts(
          mlmd_handle, execution.id
      )
      artifact_lib.update_artifacts(
          mlmd_handle,
          pending_output_artifacts,
          types.artifact.ArtifactState.ABANDONED,
      )


def _run_end_nodes(
    mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
    task_queue: tq.TaskQueue,
    pipeline_state: pstate.PipelineState,
    service_job_manager: service_jobs.ServiceJobManager,
):
  """Runs any end node that should be ran.

  Args:
    mlmd_connection_manager: Connection manager to manager multiple mlmd
      connections.
    task_queue: TaskQueue for managing tasks for nodes.
    pipeline_state: PipelineState object for this pipeline run.
    service_job_manager: Manager for service jobs. Unused but needed to
      construct a SyncPipelineTaskGenerator.
  """
  # Build some dicts and find all paired nodes
  end_nodes = []
  pipeline = pipeline_state.pipeline
  nodes = pstate.get_all_nodes(pipeline)
  node_uid_by_id = {}
  with pipeline_state:
    node_state_by_node_uid = pipeline_state.get_node_states_dict()
  for node in nodes:
    node_uid_by_id[node.node_info.id] = task_lib.NodeUid.from_node(
        pipeline, node
    )
    if not node.execution_options.HasField('resource_lifetime'):
      logging.info('Node %s has no resource lifetime', node.node_info.id)
      continue
    resource_lifetime = node.execution_options.resource_lifetime
    if resource_lifetime.HasField('lifetime_start'):
      logging.info(
          'Node %s is an end node with upstream %s',
          node.node_info.id,
          resource_lifetime.lifetime_start,
      )
      end_nodes.append(node)
  logging.info('end_nodes: %s', [n.node_info.id for n in end_nodes])
  end_nodes_to_start = []
  # Find end nodes to start, and those that are already running.
  for end_node in end_nodes:
    node_id = end_node.node_info.id

    logging.info('checking if end node %s should be started', node_id)
    end_node_state = node_state_by_node_uid[node_uid_by_id[node_id]]
    upstream_node_uid = node_uid_by_id[
        end_node.execution_options.resource_lifetime.lifetime_start
    ]
    start_node_state = node_state_by_node_uid[upstream_node_uid]
    if start_node_state.is_success() and not end_node_state.is_success():
      logging.info(
          'Node %s in state %s should be started',
          node_id,
          end_node_state.state,
      )
      end_nodes_to_start.append(end_node)
    else:
      logging.info(
          'Node %s in state %s should not be started',
          node_id,
          end_node_state.state,
      )

  logging.info(
      'Starting end nodes: %s', [n.node_info.id for n in end_nodes_to_start]
  )
  if not end_nodes_to_start:
    return
  generated_tasks = []
  generator = sync_pipeline_task_gen.SyncPipelineTaskGenerator(
      mlmd_connection_manager,
      task_queue.contains_task_id,
      service_job_manager,
  )
  for node in end_nodes_to_start:
    # We never want to crash here to wrap everything in a try/except. If we
    # are unable to generate cleanup tasks then log, mark the node as FAILED,
    # and move on.
    try:
      logging.info('generating tasks for node %s', node.node_info.id)
      tasks = generator.get_tasks_for_node(node, pipeline_state)
      generated_tasks.extend(tasks)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to generate tasks for paired end node %s: %s',
          node,
          e,
      )
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            node_uid_by_id[node.node_info.id]
        ) as node_state:
          logging.info(
              'Marking node %s as failed since we failed to generate tasks for'
              ' it during cleaup.',
              node.node_info.id,
          )
          node_state.update(
              pstate.NodeState.FAILED,
              status=status_lib.Status(
                  code=status_lib.Code.INTERNAL,
                  message=f'Unable to run end node during cleanup: {e}',
              ),
          )
        continue

  with pipeline_state:
    for task in generated_tasks:
      if isinstance(task, task_lib.UpdateNodeStateTask):
        # TODO(b/272015049): Revist how to display launched jobs
        logging.info(
            'Got update node state task for node %s, to state %s',
            task.node_uid.node_id,
            task.state,
        )
      elif isinstance(task, task_lib.ExecNodeTask):
        logging.info('Got exec task for node %s', task.node_uid.node_id)
        task_queue.enqueue(task)
      else:
        logging.error('Unsupported task: %s', task.task_id)


def _orchestrate_stop_initiated_pipeline(
    mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
    task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState,
) -> None:
  """Orchestrates stop initiated pipeline."""
  nodes_to_stop = []
  with pipeline_state:
    pipeline = pipeline_state.pipeline
    stop_reason = pipeline_state.stop_initiated_reason()
    assert stop_reason is not None
    for node in pstate.get_all_nodes(pipeline):
      node_uid = task_lib.NodeUid.from_node(pipeline, node)
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.is_stoppable():
          node_state.update(
              pstate.NodeState.STOPPING,
              # We don't use the pipeline level status as node status because
              # pipeline level status may reflect the status of another failed
              # node in the pipeline which triggered this pipeline stop
              # operation, so imputing the pipeline level status to nodes being
              # cancelled could be misleading.
              status_lib.Status(code=status_lib.Code.CANCELLED),
          )
      if node_state.state == pstate.NodeState.STOPPING:
        nodes_to_stop.append(node)

  # Issue cancellation for nodes_to_stop and gather the ones whose stopping is
  # complete.
  stopped_nodes = []
  for node in nodes_to_stop:
    if _cancel_node(
        mlmd_connection_manager.primary_mlmd_handle,
        task_queue,
        service_job_manager,
        pipeline_state,
        node,
    ):
      stopped_nodes.append(node)

  # Change the state of stopped nodes to STOPPED.
  with pipeline_state:
    for node in stopped_nodes:
      node_uid = task_lib.NodeUid.from_node(pipeline, node)
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        node_state.update(pstate.NodeState.STOPPED, node_state.status)

  logging.info('stopped nodes: %s', [n.node_info.id for n in stopped_nodes])
  # If all the nodes_to_stop have been stopped, we can update the pipeline
  # execution state.
  nodes_to_stop_ids = set(n.node_info.id for n in nodes_to_stop)
  stopped_nodes_ids = set(n.node_info.id for n in stopped_nodes)
  all_stopped = nodes_to_stop_ids == stopped_nodes_ids
  if all_stopped:
    with pipeline_state:
      # Update pipeline execution state in MLMD.
      pipeline_state.set_pipeline_execution_state(
          _mlmd_execution_code(stop_reason)
      )
      event_observer.notify(
          event_observer.PipelineFinished(
              pipeline_uid=pipeline_state.pipeline_uid,
              pipeline_state=pipeline_state,
              status=stop_reason,
          )
      )
    if any(
        n.execution_options.HasField('resource_lifetime')
        for n in pstate.get_all_nodes(pipeline_state.pipeline)
    ):
      logging.info('Pipeline has paired nodes. May launch additional jobs')
      # Note that this is a pretty hacky "best effort" attempt at cleanup, we
      # Put the ExecNodeTasks into the task_queue but do no monitoring of them,
      # and we do not support node re-try if the cleanup task fails.
      # TODO(b/272015049): If requested support retry of cleanup tasks.
      try:
        _run_end_nodes(
            mlmd_connection_manager,
            task_queue,
            pipeline_state,
            service_job_manager,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.exception('Failed to run end nodes: %s', e)
    else:
      logging.info('No paired nodes found in pipeline.')
  else:
    logging.info(
        'Not all nodes stopped! node_to_stop: %s, stopped_nodes: %s',
        nodes_to_stop_ids,
        stopped_nodes_ids,
    )


def _orchestrate_update_initiated_pipeline(
    mlmd_handle: metadata.Metadata,
    task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState,
) -> None:
  """Orchestrates an update-initiated pipeline."""
  nodes_to_stop = []
  with pipeline_state:
    update_options = pipeline_state.get_update_options()
    reload_node_ids = (
        list(update_options.reload_nodes)
        if update_options.reload_policy == update_options.PARTIAL
        else None
    )
    pipeline = pipeline_state.pipeline
    for node in pstate.get_all_nodes(pipeline):
      # TODO(b/217584342): Partial reload which excludes service nodes is not
      # fully supported in async pipelines since we don't have a mechanism to
      # reload them later for new executions.
      if (
          reload_node_ids is not None
          and node.node_info.id not in reload_node_ids
      ):
        continue
      node_uid = task_lib.NodeUid.from_node(pipeline, node)
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        if node_state.is_stoppable():
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(
                  code=status_lib.Code.CANCELLED, message=_STOPPED_BY_UPDATE
              ),
          )
      if node_state.state == pstate.NodeState.STOPPING:
        nodes_to_stop.append(node)

  # Issue cancellation for nodes_to_stop and gather the ones whose STOPPING is
  # complete.
  stopped_nodes = []
  for node in nodes_to_stop:
    if _cancel_node(
        mlmd_handle,
        task_queue,
        service_job_manager,
        pipeline_state,
        node,
    ):
      stopped_nodes.append(node)

  # Change the state of stopped nodes to STOPPED.
  with pipeline_state:
    for node in stopped_nodes:
      node_uid = task_lib.NodeUid.from_node(pipeline, node)
      with pipeline_state.node_state_update_context(node_uid) as node_state:
        node_state.update(pstate.NodeState.STOPPED, node_state.status)

  # If all the stoppable nodes have been stopped, we can update the node state
  # to STARTED.
  all_stopped = set(n.node_info.id for n in nodes_to_stop) == set(
      n.node_info.id for n in stopped_nodes
  )
  if all_stopped:
    with pipeline_state:
      pipeline = pipeline_state.pipeline
      for node in pstate.get_all_nodes(pipeline):
        # TODO(b/217584342): Partial reload which excludes service nodes is not
        # fully supported in async pipelines since we don't have a mechanism to
        # reload them later for new executions.
        if (
            reload_node_ids is not None
            and node.node_info.id not in reload_node_ids
        ):
          continue
        node_uid = task_lib.NodeUid.from_node(pipeline, node)
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          if (
              node_state.state == pstate.NodeState.STOPPED
              and node_state.status_msg == _STOPPED_BY_UPDATE
          ):
            node_state.update(pstate.NodeState.STARTED)

      pipeline_state.apply_pipeline_update()


@attr.s(auto_attribs=True, kw_only=True)
class _NodeInfo:
  """A convenience container of pipeline node and its state."""

  node: node_proto_view.NodeProtoView
  state: pstate.NodeState


def _orchestrate_active_pipeline(
    mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
    task_queue: tq.TaskQueue,
    service_job_manager: service_jobs.ServiceJobManager,
    pipeline_state: pstate.PipelineState,
) -> None:
  """Orchestrates active pipeline."""
  pipeline = pipeline_state.pipeline
  with pipeline_state:
    assert pipeline_state.is_active()
    if pipeline_state.pipeline_decode_error is not None:
      pipeline_state.initiate_stop(
          status_lib.Status(
              code=status_lib.Code.INTERNAL,
              message=(
                  'Pipeline aborted due to failure to load pipeline IR: '
                  f'{str(pipeline_state.pipeline_decode_error)}'
              ),
          )
      )
      return
    if pipeline_state.get_pipeline_execution_state() != (
        metadata_store_pb2.Execution.RUNNING
    ):
      pipeline_state.set_pipeline_execution_state(
          metadata_store_pb2.Execution.RUNNING
      )
    orchestration_options = pipeline_state.get_orchestration_options()
    logging.info('Orchestration options: %s', orchestration_options)
    deadline_secs = orchestration_options.deadline_secs
    if (
        pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC
        and deadline_secs > 0
        and time.time()
        - pipeline_state.pipeline_creation_time_secs_since_epoch()
        > deadline_secs
    ):
      logging.error(
          (
              'Aborting pipeline due to exceeding deadline (%s secs); '
              'pipeline uid: %s'
          ),
          deadline_secs,
          pipeline_state.pipeline_uid,
      )
      pipeline_state.initiate_stop(
          status_lib.Status(
              code=status_lib.Code.DEADLINE_EXCEEDED,
              message=(
                  'Pipeline aborted due to exceeding deadline '
                  f'({deadline_secs} secs)'
              ),
          )
      )
      return

  def _filter_by_state(
      node_infos: List[_NodeInfo], state_str: str
  ) -> List[_NodeInfo]:
    return [n for n in node_infos if n.state.state == state_str]

  def _filter_by_node_id(
      node_infos: List[_NodeInfo], node_id: str
  ) -> _NodeInfo:
    results = [n for n in node_infos if n.node.node_info.id == node_id]
    assert len(results) == 1
    return results[0]

  node_infos = _get_node_infos(pipeline_state)
  stopping_node_infos = _filter_by_state(node_infos, pstate.NodeState.STOPPING)

  # Tracks nodes stopped in the current iteration.
  stopped_node_infos: List[_NodeInfo] = []

  # Create cancellation tasks for nodes in state STOPPING.
  for node_info in stopping_node_infos:
    if _cancel_node(
        mlmd_connection_manager.primary_mlmd_handle,
        task_queue,
        service_job_manager,
        pipeline_state,
        node_info.node,
    ):
      stopped_node_infos.append(node_info)

  # Change the state of stopped nodes from STOPPING to STOPPED.
  if stopped_node_infos:
    with pipeline_state:
      for node_info in stopped_node_infos:
        node_uid = task_lib.NodeUid.from_node(pipeline, node_info.node)
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STOPPED, node_state.status)

  # Initialize task generator for the pipeline.
  if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
    generator = sync_pipeline_task_gen.SyncPipelineTaskGenerator(
        mlmd_connection_manager,
        task_queue.contains_task_id,
        service_job_manager,
        fail_fast=orchestration_options.fail_fast,
    )
  elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
    generator = async_pipeline_task_gen.AsyncPipelineTaskGenerator(
        mlmd_connection_manager,
        task_queue.contains_task_id,
        service_job_manager,
    )
  else:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(
            'Only SYNC and ASYNC pipeline execution modes supported; '
            f'found pipeline with execution mode: {pipeline.execution_mode}'
        ),
    )

  logging.info('Generating tasks for pipeline %s', pipeline_state.pipeline_uid)
  tasks = generator.generate(pipeline_state)
  logging.info(
      'Generated tasks for pipeline %s: %s',
      pipeline_state.pipeline_uid,
      [t.task_id for t in tasks],
  )

  # If nodes reach a terminal state, call stop_node_services for pure/mixed
  # service nodes, and cancel active executions.
  for task in tasks:
    if not isinstance(task, task_lib.UpdateNodeStateTask):
      continue
    if not (
        pstate.is_node_state_success(task.state)
        or pstate.is_node_state_failure(task.state)
    ):
      continue

    node_id = task.node_uid.node_id
    if service_job_manager.is_pure_service_node(
        pipeline_state, node_id
    ) or service_job_manager.is_mixed_service_node(pipeline_state, node_id):
      logging.info('Stopping services for node: %s', task.node_uid)
      if not service_job_manager.stop_node_services(pipeline_state, node_id):
        logging.warning(
            'Ignoring failure to stop services for node %s which is in'
            ' state %s',
            task.node_uid,
            task.state,
        )

    if pstate.is_node_state_failure(task.state):
      logging.info(
          'Canceling active executions for failed node: %s',
          task.node_uid,
      )
      node = _filter_by_node_id(node_infos, node_id).node
      active_executions = task_gen_utils.get_executions(
          mlmd_connection_manager.primary_mlmd_handle,
          node,
          additional_filters=['last_known_state IN (NEW, RUNNING)'],
      )
      _cancel_executions(
          active_executions,
          mlmd_connection_manager.primary_mlmd_handle,
          task.node_uid,
      )

  with pipeline_state:
    # Handle all the UpdateNodeStateTasks by updating node states.
    for task in tasks:
      if isinstance(task, task_lib.UpdateNodeStateTask):
        with pipeline_state.node_state_update_context(
            task.node_uid
        ) as node_state:
          node_state.update(task.state, task.status, task.backfill_token)

    tasks = [
        t for t in tasks if not isinstance(t, task_lib.UpdateNodeStateTask)
    ]
    for task in tasks:
      if isinstance(task, task_lib.ExecNodeTask):
        task_queue.enqueue(task)
      else:
        assert isinstance(task, task_lib.FinalizePipelineTask)
        assert pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC
        assert len(tasks) == 1
        if task.status.code == status_lib.Code.OK:
          logging.info(
              'Pipeline run successful; pipeline uid: %s',
              pipeline_state.pipeline_uid,
          )
        else:
          logging.info(
              'Pipeline run failed; pipeline uid: %s',
              pipeline_state.pipeline_uid,
          )
        pipeline_state.initiate_stop(task.status)


def _get_node_infos(pipeline_state: pstate.PipelineState) -> List[_NodeInfo]:
  """Returns a list of `_NodeInfo` object for each node in the pipeline."""
  nodes = pstate.get_all_nodes(pipeline_state.pipeline)
  result: List[_NodeInfo] = []
  with pipeline_state:
    for node in nodes:
      node_uid = task_lib.NodeUid.from_node(pipeline_state.pipeline, node)
      result.append(
          _NodeInfo(node=node, state=pipeline_state.get_node_state(node_uid))
      )
  return result


def _maybe_enqueue_cancellation_task(
    mlmd_handle: metadata.Metadata,
    pipeline_state: pstate.PipelineState,
    node: node_proto_view.NodeProtoView,
    task_queue: tq.TaskQueue,
) -> bool:
  """Try to cancel all active executions and enqueue cancellation task.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_state: The pipeline state of the pipeline containing the node to
      cancel.
    node: The node to cancel.
    task_queue: A `TaskQueue` instance into which any cancellation tasks will be
      enqueued.

  Returns:
    `True` if the node hasn't been stopped, and a cancellation task is enqueued.
    `False` if the node is already stopped or no cancellation is required.
  """
  executions = task_gen_utils.get_executions(
      mlmd_handle,
      node,
      additional_filters=['last_known_state IN (NEW, RUNNING)'],
  )
  pipeline = pipeline_state.pipeline
  node_uid = task_lib.NodeUid.from_node(pipeline, node)

  # Changes all NEW executions to CANCELED.
  for execution in executions:
    if execution.last_known_state == metadata_store_pb2.Execution.NEW:
      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=mlmd_handle,
          execution_id=execution.id,
          on_commit=event_observer.make_notify_execution_state_change_fn(
              node_uid
          ),
      ) as execution:
        execution.last_known_state = metadata_store_pb2.Execution.CANCELED

  # If the node has an ExecNodeTask in the task queue, issue a CancelNodeTask.
  exec_node_task_id = task_lib.exec_node_task_id_from_node(pipeline, node)
  cancel_type = task_lib.NodeCancelType.CANCEL_EXEC
  if task_queue.contains_task_id(exec_node_task_id):
    task_queue.enqueue(
        task_lib.CancelNodeTask(node_uid=node_uid, cancel_type=cancel_type)
    )
    return True

  # When the node has an active execution in MLMD but no ExecNodeTask in
  # task_queue, maybe it is because the orchestrator restarted and the
  # task_queue was clear. So, we enqueue an ExecNodeTask with cancel_type to let
  # the scheduler finish gracefully.
  exec_node_task = task_gen_utils.generate_cancel_task_from_running_execution(
      mlmd_handle, pipeline, node, executions, cancel_type=cancel_type
  )
  if exec_node_task:
    task_queue.enqueue(exec_node_task)
    return True

  return False


def _mlmd_execution_code(
    status: status_lib.Status,
) -> metadata_store_pb2.Execution.State:
  if status.code == status_lib.Code.OK:
    return metadata_store_pb2.Execution.COMPLETE
  elif status.code == status_lib.Code.CANCELLED:
    return metadata_store_pb2.Execution.CANCELED
  return metadata_store_pb2.Execution.FAILED


@dataclasses.dataclass(frozen=True)
class _MLMDProtos:
  """Represents the MLMD protos associated with an execution."""

  # Used for URI generation for internal intermediate artifacts. Also partially
  # deep copied when constructing the intermediate artifact.
  reference_artifact: metadata_store_pb2.Artifact

  # Used to verify that a user provided external URI is unqique.
  # TODO(b/299374487): Change to `list` once lowerbound Python
  # version is update to 3.9.
  intermediate_artifacts: List[metadata_store_pb2.Artifact]


def _get_mlmd_protos_for_execution(
    mlmd_handle: metadata.Metadata,
    execution_id: int,
    output_key: str,
) -> _MLMDProtos:
  """Gets MLMD protos associated with the execution ID and output key.

  Args:
    mlmd_handle: A handle to the MLMD database.
    execution_id: The execution ID.
    output_key: The output key.

  Returns:
    A _MLMDProtos struct with the MLMD protos for the reference artifact,
    intermediate artifacts, artifact type, and execution.
  """
  # Get the LineageGraph associated with the execution.
  try:
    lineage_graph = mlmd_handle.store.get_lineage_subgraph(
        query_options=metadata_store_pb2.LineageSubgraphQueryOptions(
            starting_executions=(
                metadata_store_pb2.LineageSubgraphQueryOptions.StartingNodes(
                    filter_query=f'id = {execution_id}',
                )
            ),
            max_num_hops=1,
            direction=metadata_store_pb2.LineageSubgraphQueryOptions.DOWNSTREAM,
        ),
        field_mask_paths=[
            'artifacts',
            'events',
        ],
    )
  except mlmd_errors.StatusError as e:
    raise status_lib.StatusNotOkError(code=e.error_code, message=str(e))

  output_artifact_ids = set()
  for event in lineage_graph.events:
    # We check both OUTPUT and PENDING_OUTPUT state because the REFERENCE
    # artifact will have event type PENDING_OUTPUT, but LIVE intermediate
    # artifacts will have event type OUTPUT.
    if event_lib.contains_key(event, output_key) and event.type in [
        metadata_store_pb2.Event.PENDING_OUTPUT,
        metadata_store_pb2.Event.OUTPUT,
    ]:
      output_artifact_ids.add(event.artifact_id)
  output_artifacts = [
      a for a in lineage_graph.artifacts if a.id in output_artifact_ids
  ]

  # Find the REFERENCE and LIVE artifacts in the subgraph.
  reference_artifact = None
  intermediate_artifacts = []
  for artifact in output_artifacts:
    if artifact.state == metadata_store_pb2.Artifact.State.REFERENCE:
      if reference_artifact is not None:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.ALREADY_EXISTS,
            message=(
                'Found multiple REFERENCE Artifacts with output_key '
                f'{output_key} for execution_id {execution_id}.'
            ),
        )
      reference_artifact = artifact

    elif artifact.state == metadata_store_pb2.Artifact.State.LIVE:
      intermediate_artifacts.append(artifact)

  if reference_artifact is None:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.NOT_FOUND,
        message=(
            f'REFERENCE Artifact with output_key {output_key} for '
            f'execution_id {execution_id} not found.'
        ),
    )

  return _MLMDProtos(
      reference_artifact=reference_artifact,
      intermediate_artifacts=intermediate_artifacts,
  )


def _generate_reference_uri_subdir(
    reference_artifact_uri: str,
) -> str:
  """Generates and returns the URI for the intermediate artifact."""
  # TODO(b/285399450): Properly handle ValueArtifacts, which have a uri of
  # a file, e.g. some/uri/value instead of a directory.

  now = datetime.datetime.now(datetime.timezone.utc)
  # The subdirectory will be intermediate_artifact_YYYYMMDD_HHMMSS_FFFFFF.
  subdirectory = now.strftime(f'{constants.PREFIX}_%Y%m%d_%H%M%S_%f')

  # Return the intermediate artifact URI.
  return os.path.join(reference_artifact_uri, subdirectory)


# The decorator applies the same lock used in OrchestratorServicer.
@_pipeline_op()
def publish_intermediate_artifact(
    mlmd_handle: metadata.Metadata,
    execution_id: int,
    output_key: str,
    properties: Optional[Dict[str, metadata_store_pb2.Value]],
    custom_properties: Optional[Dict[str, metadata_store_pb2.Value]],
    external_uri: Optional[str] = None,
    temp_uri: Optional[str] = None,
) -> metadata_store_pb2.Artifact:
  """Publishes an intermediate artifact.

  Args:
    mlmd_handle: A handle to the MLMD database.
    execution_id: The ID of the execution which generates the artifact.
    output_key: The output key of the artifact.
    properties: Properties of the artifact.
    custom_properties: Custom properties of the artifact.
    external_uri: The external URI provided by the user. Exactly one of
      external_uri and temp_uri must be set.
    temp_uri: Temp URI generated internally by Tflex. Exactly one of
      external_uri and temp_uri must be set.

  Returns:
    The published intermediate Artifact proto.
  """
  # Check that a REFERENCE artifact corresponding to the output key and
  # execution ID exists.
  mlmd_protos = _get_mlmd_protos_for_execution(
      mlmd_handle, execution_id, output_key
  )

  if external_uri:
    # The final URI for the intermediate artifact is an external URI.
    final_uri = external_uri

    # Verify that an external artifact with the same URI has not already been
    # published.
    for artifact in mlmd_protos.intermediate_artifacts:
      if artifact.uri == final_uri:
        raise status_lib.StatusNotOkError(
            code=status_lib.Code.ALREADY_EXISTS,
            message=(
                f'Artifact with URI {final_uri} has already been published: '
                f'{artifact}'
            ),
        )
  elif temp_uri:
    # The final URI for the intermediate artifact is a subdirectory of the
    # REFERENCE artifact's URI.
    final_uri = _generate_reference_uri_subdir(
        mlmd_protos.reference_artifact.uri,
    )

    try:
      fileio.rename(temp_uri, final_uri)
    except filesystem.NotFoundError as e:
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ABORTED, message=str(e)
      )
    logging.info(
        'Moved temporary URI %s contents to final URI %s',
        temp_uri,
        final_uri,
    )
  else:
    raise status_lib.StatusNotOkError(
        code=status_lib.Code.INVALID_ARGUMENT,
        message='Neither external_uri nor temp_uri was provided.',
    )

  # Build the intermediate artifact object. We set its state to LIVE, so that
  # it can be immediately consumed.
  intermediate_artifact = metadata_store_pb2.Artifact()
  intermediate_artifact.CopyFrom(mlmd_protos.reference_artifact)
  intermediate_artifact.uri = final_uri
  intermediate_artifact.state = metadata_store_pb2.Artifact.State.LIVE
  intermediate_artifact.ClearField('id')
  intermediate_artifact.ClearField('create_time_since_epoch')
  intermediate_artifact.ClearField('last_update_time_since_epoch')

  # Copy any new properties/custom properties for the artifact.
  if properties:
    for key, value in properties.items():
      intermediate_artifact.properties[key].CopyFrom(value)
  if custom_properties:
    for key, value in custom_properties.items():
      intermediate_artifact.custom_properties[key].CopyFrom(value)

  try:
    contexts = mlmd_handle.store.get_contexts_by_execution(execution_id)
    event = event_lib.generate_event(
        event_type=metadata_store_pb2.Event.OUTPUT,
        key=output_key,
        # We intentionally start the OUTPUT Event at index at 0, even though
        # there is a PENDING_OUTPUT Event with index 0 associated with the
        # REFERENCE artifact.
        index=len(mlmd_protos.intermediate_artifacts),
    )
    # TODO(b/262040844): Instead of directly using the context manager here, we
    # should consider creating and using wrapper functions.
    with mlmd_state.evict_from_cache(execution_id):
      [execution] = mlmd_handle.store.get_executions_by_id([execution_id])
      # Link the Execution to the Artifact with an OUTPUT Event edge.
      mlmd_handle.store.put_execution(
          execution=execution,
          artifact_and_events=[(intermediate_artifact, event)],
          contexts=contexts,
          reuse_context_if_already_exist=True,
          reuse_artifact_if_already_exist_by_external_id=True,
          # Intermediate artifacts are published after the execution is created.
          # We need to set force_update_time to True, to ensuer
          # last_update_time_since_epoch is updated whenevery we publish new
          # intermediate artifacts.
          force_update_time=True,
      )

  except mlmd_errors.StatusError as e:
    raise status_lib.StatusNotOkError(code=e.error_code, message=str(e))

  logging.info('Published intermediate artifact: %s', intermediate_artifact)
  return intermediate_artifact
