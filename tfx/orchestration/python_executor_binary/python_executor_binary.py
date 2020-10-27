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
r"""This module defines the binary used to run PythonExecutorOperator in Tflex.

Example:

python_executor_binary
--tfx_execution_info_b64=ChcKEAgBEgxnZW5lcmljX3R5cGUSA2ZvbyoPCg0KAnAxEgcKBRoDYmFy
\
--tfx_python_class_executor_spec_b64=ChcKEAgBEgxnZW5lcmljX3R5cGUSA2ZvbyoPCg0KAnAxEgcKBRoDYmFy
\
--alsologtostderr

This binary is intended to be called by the Tflex IR based launcher and should
not be called directly.

"""
import base64

from absl import app
from absl import flags
from absl import logging

from tfx.dsl.io import fileio
from tfx.orchestration.portable import python_executor_operator
from tfx.orchestration.python_executor_binary import python_executor_binary_utils
from tfx.proto.orchestration import executable_spec_pb2
from google.protobuf import text_format

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'tfx_execution_info_b64', None, 'base64 encoded binary '
    'tfx.orchestration.PythonExecutorExecutionInfo proto')
flags.DEFINE_string(
    'tfx_python_class_executor_spec_b64', None,
    'tfx.orchestration.executable_spec.PythonClassExecutableSpec proto')


def main(_):

  flags.mark_flag_as_required('tfx_execution_info_b64')
  flags.mark_flag_as_required('tfx_python_class_executor_spec_b64')

  execution_info = python_executor_binary_utils.deserialize_execution_info(
      FLAGS.tfx_execution_info_b64)
  python_class_executor_spec = executable_spec_pb2.PythonClassExecutableSpec.FromString(
      base64.b64decode(FLAGS.tfx_python_class_executor_spec_b64))

  logging.info('execution_info = %s\n',
               text_format.MessageToString(execution_info))
  logging.info('python_class_executor_spec = %s\n',
               text_format.MessageToString(python_class_executor_spec))

  operator = python_executor_operator.PythonExecutorOperator(
      python_class_executor_spec)
  run_result = operator.run_executor(execution_info)

  if run_result:
    with fileio.open(execution_info.executor_output_uri, 'wb') as f:
      f.write(run_result.SerializeToString())


if __name__ == '__main__':
  app.run(main)
