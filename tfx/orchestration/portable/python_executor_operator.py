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

from typing import cast, Optional

from tfx.orchestration.portable import base_executor_operator
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import import_utils

from google.protobuf import message


# WARNING: This class is under development and is not ready to be used yet.
# TODO(b/150979622): Implement this class.
class PythonExecutorOperator(base_executor_operator.BaseExecutorOperator):
  """PythonExecutorOperator handles python class based executor's init and execution."""

  SUPPORTED_EXECUTOR_SPEC_TYPE = [
      pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec
  ]

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
    executor_cls = import_utils.import_class_by_path(
        python_class_executor_spec.class_path)
    self._executor = executor_cls()

  def run_executor(
      self, execution_info: base_executor_operator.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    """Invokers executors given input from the Launcher.

    Args:
      execution_info: A wrapper of the details of this execution.

    Returns:
      The output from executor.
    """
    return execution_result_pb2.ExecutorOutput()
