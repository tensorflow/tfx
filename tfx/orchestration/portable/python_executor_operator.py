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
"""Base class to define how to operator an executor."""

from typing import Any, Dict, List, Optional, cast

from absl import logging
import tensorflow as tf
from tfx import types
from tfx.components.base import base_executor
from tfx.orchestration.portable import base_executor_operator
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import import_utils

from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

_STATEFUL_WORKING_DIR = 'stateful_working_dir'


def _populate_output_artifact(
    executor_output: execution_result_pb2.ExecutorOutput,
    output_dict: Dict[str, List[types.Artifact]]):
  """Populate output_dict to executor_output."""
  for key, artifact_list in output_dict.items():
    artifacts = execution_result_pb2.ExecutorOutput.ArtifactList()
    for artifact in artifact_list:
      artifacts.artifacts.append(artifact.mlmd_artifact)
    executor_output.output_artifacts[key].CopyFrom(artifacts)


def _populate_exec_properties(
    executor_output: execution_result_pb2.ExecutorOutput,
    exec_properties: Dict[str, Any]):
  """Populate exec_properties to executor_output."""
  for key, value in exec_properties.items():
    v = metadata_store_pb2.Value()
    if isinstance(value, str):
      v.string_value = value
    elif isinstance(value, int):
      v.int_value = value
    elif isinstance(value, float):
      v.double_value = value
    else:
      logging.info(
          'Value type %s of key %s in exec_properties is not '
          'supported, going to drop it', type(value), key)
      continue
    executor_output.execution_properties[key].CopyFrom(v)


class PythonExecutorOperator(base_executor_operator.BaseExecutorOperator):
  """PythonExecutorOperator handles python class based executor's init and execution."""

  SUPPORTED_EXECUTOR_SPEC_TYPE = [executable_spec_pb2.PythonClassExecutableSpec]
  SUPPORTED_PLATFORM_SPEC_TYPE = []

  def __init__(self,
               executor_spec: message.Message,
               platform_spec: Optional[message.Message] = None):
    """Initialize an PythonExecutorOperator.

    Args:
      executor_spec: The specification of how to initialize the executor.
      platform_spec: The specification of how to allocate resource for the
        executor.
    """
    # Python exectors run locally, so platform_spec is not used.
    del platform_spec
    super(PythonExecutorOperator, self).__init__(executor_spec)
    python_class_executor_spec = cast(
        pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec, self._executor_spec)
    self._executor_cls = import_utils.import_class_by_path(
        python_class_executor_spec.class_path)

  def run_executor(
      self, execution_info: base_executor_operator.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    """Invokers executors given input from the Launcher.

    Args:
      execution_info: A wrapper of the details of this execution.

    Returns:
      The output from executor.
    """
    # TODO(b/162980675): Set arguments for Beam when it is available.
    context = base_executor.BaseExecutor.Context(
        executor_output_uri=execution_info.executor_output_uri,
        stateful_working_dir=execution_info.stateful_working_dir)
    executor = self._executor_cls(context=context)

    result = executor.Do(execution_info.input_dict, execution_info.output_dict,
                         execution_info.exec_properties)
    if not result:
      # If result is not returned from the Do function, then try to
      # read if from the executor_output_uri.
      try:
        with tf.io.gfile.GFile(execution_info.executor_output_uri, 'rb') as f:
          result = execution_result_pb2.ExecutorOutput.FromString(
              f.read())
      except tf.errors.NotFoundError:
        # Old style TFX executor doesn't return executor_output, but modify
        # output_dict and exec_properties in place. For backward compatibility,
        # we use their executor_output and exec_properties to construct
        # ExecutorOutput.
        result = execution_result_pb2.ExecutorOutput()
        _populate_output_artifact(result, execution_info.output_dict)
        _populate_exec_properties(result, execution_info.exec_properties)
    return result
