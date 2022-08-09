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
r"""This module defines the entrypoint for the PythonExecutorOperator in TFX.

This library is intended to serve as the entrypoint for a binary that packages
the python executors in a pipeline. The resulting binary is called by the TFX
launcher and should not be called directly.
"""
from typing import Union, cast

from absl import flags
from absl import logging
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.portable import beam_executor_operator
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import python_driver_operator
from tfx.orchestration.portable import python_executor_operator
from tfx.orchestration.python_execution_binary import python_execution_binary_utils
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.utils import import_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS

EXECUTION_INVOCATION_FLAG = flags.DEFINE_string(
    'tfx_execution_info_b64', None, 'url safe base64 encoded binary '
    'tfx.orchestration.ExecutionInvocation proto')
EXECUTABLE_SPEC_FLAG = flags.DEFINE_string(
    'tfx_python_class_executable_spec_b64', None,
    'tfx.orchestration.executable_spec.PythonClassExecutableSpec proto')
BEAM_EXECUTABLE_SPEC_FLAG = flags.DEFINE_string(
    'tfx_beam_executable_spec_b64', None,
    'tfx.orchestration.executable_spec.BeamExecutableSpec proto')
MLMD_CONNECTION_CONFIG_FLAG = flags.DEFINE_string(
    'tfx_mlmd_connection_config_b64', None,
    'wrapper proto containing MLMD connection config. If being set, this'
    'indicates a driver execution')


def _import_class_path(
    executable_spec: Union[executable_spec_pb2.PythonClassExecutableSpec,
                           executable_spec_pb2.BeamExecutableSpec],):
  """Import the class path from Python or Beam executor spec."""
  if isinstance(executable_spec, executable_spec_pb2.BeamExecutableSpec):
    beam_executor_spec = cast(executable_spec_pb2.BeamExecutableSpec,
                              executable_spec)
    import_utils.import_class_by_path(
        beam_executor_spec.python_executor_spec.class_path)
  else:
    python_class_executor_spec = cast(
        executable_spec_pb2.PythonClassExecutableSpec, executable_spec)
    import_utils.import_class_by_path(python_class_executor_spec.class_path)


def _run_executor(
    executable_spec: Union[executable_spec_pb2.PythonClassExecutableSpec,
                           executable_spec_pb2.BeamExecutableSpec],
    execution_info: data_types.ExecutionInfo,
) -> execution_result_pb2.ExecutorOutput:
  """Run python or Beam executor operator."""
  if isinstance(executable_spec, executable_spec_pb2.BeamExecutableSpec):
    operator = beam_executor_operator.BeamExecutorOperator(executable_spec)
  else:
    operator = python_executor_operator.PythonExecutorOperator(executable_spec)
  return operator.run_executor(execution_info)


def _run_driver(
    executable_spec: Union[executable_spec_pb2.PythonClassExecutableSpec,
                           executable_spec_pb2.BeamExecutableSpec],
    mlmd_connection_config: metadata.ConnectionConfigType,
    execution_info: data_types.ExecutionInfo) -> driver_output_pb2.DriverOutput:
  operator = python_driver_operator.PythonDriverOperator(
      executable_spec, metadata.Metadata(mlmd_connection_config))
  return operator.run_driver(execution_info)


def main(_):
  flags.mark_flag_as_required(EXECUTION_INVOCATION_FLAG.name)
  flags.mark_flags_as_mutual_exclusive(
      (EXECUTABLE_SPEC_FLAG.name, BEAM_EXECUTABLE_SPEC_FLAG.name),
      required=True)

  deserialized_executable_spec = None
  if BEAM_EXECUTABLE_SPEC_FLAG.value is not None:
    deserialized_executable_spec = (
        python_execution_binary_utils.deserialize_executable_spec(
            BEAM_EXECUTABLE_SPEC_FLAG.value, with_beam=True))
  else:
    deserialized_executable_spec = (
        python_execution_binary_utils.deserialize_executable_spec(
            EXECUTABLE_SPEC_FLAG.value, with_beam=False))
  # Eagerly import class path from executable spec such that all artifact
  # references are resolved.
  _import_class_path(deserialized_executable_spec)
  execution_info = python_execution_binary_utils.deserialize_execution_info(
      EXECUTION_INVOCATION_FLAG.value)
  logging.info('execution_info = %r\n', execution_info)
  logging.info('executable_spec = %s\n',
               text_format.MessageToString(deserialized_executable_spec))

  # MLMD connection config being set indicates a driver execution instead of an
  # executor execution as accessing MLMD is not supported for executors.
  if MLMD_CONNECTION_CONFIG_FLAG.value:
    mlmd_connection_config = (
        python_execution_binary_utils.deserialize_mlmd_connection_config(
            MLMD_CONNECTION_CONFIG_FLAG.value))
    run_result = _run_driver(deserialized_executable_spec,
                             mlmd_connection_config, execution_info)
  else:
    run_result = _run_executor(
        deserialized_executable_spec, execution_info)

  if run_result:
    with fileio.open(execution_info.execution_output_uri, 'wb') as f:
      f.write(run_result.SerializeToString())
