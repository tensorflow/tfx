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

from typing import Union

from absl import flags
from absl import logging
from tfx.orchestration.python_execution_binary import python_execution_binary_utils
from tfx.orchestration.python_execution_binary import python_execution_lib
from tfx.orchestration.python_execution_binary import system_flags
from tfx.proto.orchestration import executable_spec_pb2
from tfx.utils import import_utils

from google.protobuf import text_format

FLAGS = flags.FLAGS

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


_PythonClassExecutableSpec = executable_spec_pb2.PythonClassExecutableSpec
_BeamExecutableSpec = executable_spec_pb2.BeamExecutableSpec


def _import_class_path(
    executable_spec: Union[_PythonClassExecutableSpec, _BeamExecutableSpec],
):
  """Import the class path from Python or Beam executor spec."""
  if isinstance(executable_spec, _BeamExecutableSpec):
    import_utils.import_class_by_path(
        executable_spec.python_executor_spec.class_path
    )
  elif isinstance(executable_spec, _PythonClassExecutableSpec):
    import_utils.import_class_by_path(executable_spec.class_path)
  else:
    raise ValueError(
        f'Executable spec type {type(executable_spec)} is not supported.'
    )


def main(_):
  flags.mark_flags_as_mutual_exclusive(
      (EXECUTABLE_SPEC_FLAG.name, BEAM_EXECUTABLE_SPEC_FLAG.name),
      required=True)

  if BEAM_EXECUTABLE_SPEC_FLAG.value is not None:
    executable_spec = python_execution_binary_utils.deserialize_executable_spec(
        BEAM_EXECUTABLE_SPEC_FLAG.value, with_beam=True
    )
  else:
    executable_spec = python_execution_binary_utils.deserialize_executable_spec(
        EXECUTABLE_SPEC_FLAG.value, with_beam=False
    )
  # Eagerly import class path from executable spec such that all artifact
  # references are resolved. This should come before parsing execution_info.
  _import_class_path(executable_spec)
  execution_info = system_flags.parse_execution_info()
  logging.info('execution_info = %r\n', execution_info)
  logging.info(
      'executable_spec = %s\n', text_format.MessageToString(executable_spec)
  )

  if MLMD_CONNECTION_CONFIG_FLAG.value:
    mlmd_connection_config = (
        python_execution_binary_utils.deserialize_mlmd_connection_config(
            MLMD_CONNECTION_CONFIG_FLAG.value
        )
    )
  else:
    mlmd_connection_config = None

  python_execution_lib.run(
      executable_spec, execution_info, mlmd_connection_config
  )
