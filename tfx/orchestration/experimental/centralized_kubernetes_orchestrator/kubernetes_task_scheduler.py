# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Kubernetes Task Scheduler.

First, unpack the deployment config in the given pipeline to obtain an Any type
of executor spec. Since it is an optional value, first check if it’s
None, and proceed to check its type. If it’s either of PythonClassExecutableSpec
or BeamExecutableSpec, obtain executable spec by unpacking executable Any type.

Then, obtain execution invocation given the pipeline, task, and the node.
Convert execution invocation to execution info, by using from_proto
method in ExecutionInfo class. Finally, return the result of run method in the
Kubernetes runner class, passing the obtained execution info and executable
spec.
"""
import threading

from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator import kubernetes_job_runner
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib


def _create_execution_invocation_proto(
    pipeline: pipeline_pb2.Pipeline, task: task_lib.ExecNodeTask,
    node: pipeline_pb2.PipelineNode
) -> execution_invocation_pb2.ExecutionInvocation:
  """Creates an ExecutionInvocation proto with some initial info."""

  return execution_invocation_pb2.ExecutionInvocation(
      execution_properties=(data_types_utils.build_metadata_value_dict(
          task.exec_properties)),
      execution_properties_with_schema=(
          data_types_utils.build_pipeline_value_dict(task.exec_properties)),
      output_metadata_uri=task.executor_output_uri,
      input_dict=data_types_utils.build_artifact_struct_dict(
          task.input_artifacts),
      output_dict=data_types_utils.build_artifact_struct_dict(
          task.output_artifacts),
      stateful_working_dir=task.stateful_working_dir,
      tmp_dir=task.tmp_dir,
      pipeline_info=pipeline.pipeline_info,
      pipeline_node=node,
      execution_id=task.execution_id,
      pipeline_run_id=pipeline.runtime_spec.pipeline_run_id.field_value
      .string_value)


def _get_pipeline_node(pipeline: pipeline_pb2.Pipeline,
                       node_id: str) -> pipeline_pb2.PipelineNode:
  """Gets corresponding pipeline node from IR given the node_id."""
  for node in pipeline.nodes:
    if node.pipeline_node and (node.pipeline_node.node_info.id == node_id):
      return node.pipeline_node
  raise status_lib.StatusNotOkError(
      code=status_lib.Code.INVALID_ARGUMENT,
      message=f'Failed to find corresponding node in the IR, node id: {node_id}'
  )


class KubernetesTaskScheduler(
    task_scheduler.TaskScheduler[task_lib.ExecNodeTask]):
  """Implementation of Kubernetes Task Scheduler."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: task_lib.ExecNodeTask):
    super().__init__(mlmd_handle, pipeline, task)
    self._cancel = threading.Event()
    if task.cancel_type:
      self._cancel.set()
    # TODO(b/240237394): pass tfx_image, job_prefix, container_name as
    # arguments.
    self._runner = kubernetes_job_runner.KubernetesJobRunner(
        tfx_image='',  # You need to set tfx_image with your image.
        job_prefix='sample-job',
        container_name='centralized-orchestrator')

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    """Retreive Executable Spec and Execution Info for run."""
    depl_config = pipeline_pb2.IntermediateDeploymentConfig()
    self.pipeline.deployment_config.Unpack(depl_config)
    executor_spec_any = depl_config.executor_specs.get(
        self.task.node_uid.node_id)

    if not executor_spec_any:
      return task_scheduler.TaskSchedulerResult(
          status=status_lib.Status(
              code=status_lib.Code.INVALID_ARGUMENT,
              message='Unknown executable spec type.'))

    if executor_spec_any.Is(
        executable_spec_pb2.PythonClassExecutableSpec.DESCRIPTOR):
      executable_spec = executable_spec_pb2.PythonClassExecutableSpec()
      executor_spec_any.Unpack(executable_spec)
    elif executor_spec_any.Is(
        executable_spec_pb2.BeamExecutableSpec.DESCRIPTOR):
      executable_spec = executable_spec_pb2.BeamExecutableSpec()
      executor_spec_any.Unpack(executable_spec)
    else:
      return task_scheduler.TaskSchedulerResult(
          status=status_lib.Status(
              code=status_lib.Code.INVALID_ARGUMENT,
              message='Unknown executable spec type.'))

    node = _get_pipeline_node(self.pipeline, self.task.node_uid.node_id)
    execution_invocation = _create_execution_invocation_proto(
        self.pipeline, self.task, node)
    execution_info = data_types.ExecutionInfo.from_proto(execution_invocation)

    return self._runner.run(execution_info, executable_spec)

  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    # TODO(b/240237394): implement method.
    pass
