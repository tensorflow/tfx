# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Executor operator dispatcher used by entrypoint.py."""

from typing import Union

from tfx.orchestration.portable import beam_executor_operator
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import python_executor_operator
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2


def run_executor(
    executable_spec: Union[
        executable_spec_pb2.PythonClassExecutableSpec,
        executable_spec_pb2.BeamExecutableSpec,
    ],
    execution_info: data_types.ExecutionInfo,
) -> execution_result_pb2.ExecutorOutput:
  """Run python or Beam executor operator."""
  if isinstance(executable_spec, executable_spec_pb2.BeamExecutableSpec):
    operator = beam_executor_operator.BeamExecutorOperator(executable_spec)
  else:
    operator = python_executor_operator.PythonExecutorOperator(executable_spec)
  return operator.run_executor(execution_info)
