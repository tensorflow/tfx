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

from absl import flags
from absl import logging
from tfx.orchestration.python_execution_binary import python_execution_binary_utils
from tfx.orchestration.python_execution_binary import python_execution_lib
from tfx.orchestration.python_execution_binary import system_flags

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


def main(_):
  mutually_exclusive = (EXECUTABLE_SPEC_FLAG.present) ^ (
      BEAM_EXECUTABLE_SPEC_FLAG.present
  )
  if not mutually_exclusive:
    raise ValueError(
        f'Exactly one of {EXECUTABLE_SPEC_FLAG.name} and'
        f' {BEAM_EXECUTABLE_SPEC_FLAG.name} must be set.'
    )

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
  python_execution_binary_utils.import_class_path(executable_spec)
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
