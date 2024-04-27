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
"""Task class and related functionality.

Task instructs the work to be performed. A task is typically generated by the
core task generation loop based on the state of MLMD db.
"""

import abc
import enum
from typing import Any, Dict, Hashable, List, Optional, Sequence, Type, TypeVar

import attr
from tfx import types
from tfx.orchestration import node_proto_view
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from tfx.utils import tracecontext_pb2
from ml_metadata.proto import metadata_store_pb2


# Keep Any import for OSS.
_ = Any


@attr.s(auto_attribs=True, frozen=True)
class PipelineUid:
  """Uniquely identifies a pipeline among pipelines being actively orchestrated.

  Recommended to use `from_pipeline` or `from_pipeline_id_and_run_id` class
  methods to create `PipelineUid` objects as they correctly account for
  concurrent pipeline runs mode.

  Attributes:
    pipeline_id: Id of the pipeline containing the node. Corresponds to
      `Pipeline.pipeline_info.id` in the pipeline IR.
    pipeline_run_id: Run identifier for the pipeline if one is provided.
  """
  pipeline_id: str
  pipeline_run_id: Optional[str] = None

  @classmethod
  def from_pipeline(cls: Type['PipelineUid'],
                    pipeline: pipeline_pb2.Pipeline) -> 'PipelineUid':
    """Creates a PipelineUid object given a pipeline IR."""
    if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
      pipeline_run_id = (
          pipeline.runtime_spec.pipeline_run_id.field_value.string_value
      )
      if not pipeline_run_id:
        raise ValueError(
            'pipeline_run_id unexpectedly missing for a sync pipeline.')
    else:
      pipeline_run_id = None

    return cls(
        pipeline_id=pipeline.pipeline_info.id, pipeline_run_id=pipeline_run_id)

  @classmethod
  def from_pipeline_id_and_run_id(
      cls: Type['PipelineUid'], pipeline_id: str,
      pipeline_run_id: Optional[str]) -> 'PipelineUid':
    return cls(pipeline_id=pipeline_id, pipeline_run_id=pipeline_run_id or None)


@attr.s(auto_attribs=True, frozen=True)
class NodeUid:
  """Uniquely identifies a node across all pipelines being actively orchestrated.

  Attributes:
    pipeline_uid: The pipeline UID.
    node_id: Node id. Corresponds to `PipelineNode.node_info.id` in the pipeline
      IR.
  """
  pipeline_uid: PipelineUid
  node_id: str

  @classmethod
  def from_node(cls: Type['NodeUid'], pipeline: pipeline_pb2.Pipeline,
                node: node_proto_view.NodeProtoView) -> 'NodeUid':
    return cls(
        pipeline_uid=PipelineUid.from_pipeline(pipeline),
        node_id=node.node_info.id)


# Task id can be any hashable type.
TaskId = TypeVar('TaskId', bound=Hashable)

_TaskT = TypeVar('_TaskT', bound='Task')


class Task(abc.ABC):
  """Task instructs the work to be performed."""

  @property
  @abc.abstractmethod
  def task_id(self) -> TaskId:
    """Returns a unique identifier for this task.

    The concrete implementation must ensure that the returned task id is unique
    across all task types.
    """

  @classmethod
  def task_type_id(cls: Type[_TaskT]) -> str:
    """Returns task type id."""
    return cls.__name__


@attr.s(auto_attribs=True, frozen=True)
class CancelTask(Task):
  """Base class for cancellation task types."""


@enum.unique
class NodeCancelType(enum.Enum):
  # The node is being cancelled with no intention to reuse the same execution.
  CANCEL_EXEC = 1


@attr.s(auto_attribs=True, frozen=True)
class ExecNodeTask(Task):
  """Task to instruct execution of a node in the pipeline.

  Attributes:
    node_uid: Uid of the node to be executed.
    execution_id: Id of the MLMD execution associated with the current node.
    contexts: List of contexts associated with the execution.
    exec_properties: Execution properties of the execution.
    input_artifacts: Input artifacts dict.
    output_artifacts: Output artifacts dict.
    executor_output_uri: URI for the executor output.
    stateful_working_dir: Working directory for the node execution.
    tmp_dir: Temporary directory for the node execution.
    pipeline: The pipeline IR proto containing the node to be executed.
    cancel_type: Indicates whether this is a cancelled execution, and the type
      of the cancellation. The task scheduler is expected to gracefully exit
      after doing any necessary cleanup.
    trace_parent_proto: An optional trace context proto of which task scheduler
      will add a child trace context span.
  """
  node_uid: NodeUid
  execution_id: int
  contexts: Sequence[metadata_store_pb2.Context]
  exec_properties: Dict[str, types.ExecPropertyTypes]
  input_artifacts: Dict[str, List[types.Artifact]]
  output_artifacts: Dict[str, List[types.Artifact]]
  executor_output_uri: str
  stateful_working_dir: str
  tmp_dir: str
  pipeline: pipeline_pb2.Pipeline
  cancel_type: Optional[NodeCancelType] = None
  trace_parent_proto: Optional[tracecontext_pb2.TraceContextProto] = None

  @property
  def task_id(self) -> TaskId:
    return _exec_node_task_id(self.task_type_id(), self.node_uid)

  def get_node(self) -> node_proto_view.NodeProtoView:
    for pipeline_or_node in self.pipeline.nodes:
      view = node_proto_view.get_view(pipeline_or_node)
      if view.node_info.id == self.node_uid.node_id:
        return view
    raise ValueError(
        f'Node not found in pipeline IR; node uid: {self.node_uid}')


@attr.s(auto_attribs=True, frozen=True)
class CancelNodeTask(CancelTask):
  """Task to instruct cancellation of an ongoing node execution.

  Attributes:
    node_uid: Uid of the node to be cancelled.
    cancel_type: Indicates the type of this cancellation.
  """
  node_uid: NodeUid
  cancel_type: NodeCancelType = NodeCancelType.CANCEL_EXEC

  @property
  def task_id(self) -> TaskId:
    return (self.task_type_id(), self.node_uid)


@attr.s(auto_attribs=True, frozen=True)
class FinalizePipelineTask(Task):
  """Task to instruct finalizing a pipeline run."""
  pipeline_uid: PipelineUid
  status: status_lib.Status

  @property
  def task_id(self) -> TaskId:
    return (self.task_type_id(), self.pipeline_uid)


@attr.s(auto_attribs=True, frozen=True)
class UpdateNodeStateTask(Task):
  """Task to instruct updating node states.

  This is useful for task generators to defer actually updating node states in
  MLMD to the caller, where node state updates can be bundled together with
  other pipeline state changes and committed to MLMD in a single transaction for
  efficiency.
  """
  node_uid: NodeUid
  state: str
  status: Optional[status_lib.Status] = None
  backfill_token: str = ''

  @property
  def task_id(self) -> TaskId:
    return (self.task_type_id(), self.node_uid)


def exec_node_task_id_from_node(pipeline: pipeline_pb2.Pipeline,
                                node: node_proto_view.NodeProtoView) -> TaskId:
  """Returns task id of an `ExecNodeTask` from pipeline and node."""
  return _exec_node_task_id(ExecNodeTask.task_type_id(),
                            NodeUid.from_node(pipeline, node))


def _exec_node_task_id(task_type_id: str, node_uid: NodeUid) -> TaskId:
  return (task_type_id, node_uid)
