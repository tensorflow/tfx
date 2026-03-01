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
"""Library for executing Python executables."""
from typing import Optional, Union

from absl import logging
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import python_driver_operator
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import executable_spec_pb2

from tfx.orchestration.python_execution_binary import python_executor_operator_dispatcher


_PythonClassExecutableSpec = executable_spec_pb2.PythonClassExecutableSpec
_BeamExecutableSpec = executable_spec_pb2.BeamExecutableSpec


def _run_driver(
    executable_spec: Union[_PythonClassExecutableSpec, _BeamExecutableSpec],
    mlmd_connection_config: metadata.ConnectionConfigType,
    execution_info: data_types.ExecutionInfo,
) -> driver_output_pb2.DriverOutput:
  operator = python_driver_operator.PythonDriverOperator(
      executable_spec, metadata.Metadata(mlmd_connection_config))
  return operator.run_driver(execution_info)


def run_python_custom_component(
    executable_spec: Union[_PythonClassExecutableSpec, _BeamExecutableSpec],
    execution_info: data_types.ExecutionInfo,
    mlmd_connection_config: Optional[metadata.ConnectionConfigType] = None,
) -> None:
  """Run Python custom component declared with @component decorator."""
  # MLMD connection config being set indicates a driver execution instead of an
  # executor execution as accessing MLMD is not supported for executors.
  if mlmd_connection_config:
    run_result = _run_driver(
        executable_spec, mlmd_connection_config, execution_info
    )
  else:
    run_result = python_executor_operator_dispatcher.run_executor(
        executable_spec, execution_info
    )

  if run_result:
    with fileio.open(execution_info.execution_output_uri, 'wb') as f:
      f.write(run_result.SerializeToString())


def run(
    executable_spec: Union[_PythonClassExecutableSpec, _BeamExecutableSpec],
    execution_info: data_types.ExecutionInfo,
    mlmd_connection_config: Optional[metadata.ConnectionConfigType] = None,
) -> None:
  """Run Python executable."""
  logging.info('Executing Python custom component')
  run_python_custom_component(
      executable_spec, execution_info, mlmd_connection_config
  )
