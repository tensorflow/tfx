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
"""Tests for tfx.orchestration.portable.python_executor_operator."""

import os
from typing import Any, Dict, List, Text

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import python_executor_operator
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format


class InprocessExecutor(base_executor.BaseExecutor):
  """A Fake in-process executor what returns execution result."""

  def Do(
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any]) -> execution_result_pb2.ExecutorOutput:
    executor_output = execution_result_pb2.ExecutorOutput()
    python_executor_operator._populate_output_artifact(executor_output,
                                                       output_dict)
    return executor_output


class NotInprocessExecutor(base_executor.BaseExecutor):
  """A Fake not-in-process executor what writes execution result to executor_output_uri."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    executor_output = execution_result_pb2.ExecutorOutput()
    python_executor_operator._populate_output_artifact(executor_output,
                                                       output_dict)
    with fileio.open(self._context.executor_output_uri, 'wb') as f:
      f.write(executor_output.SerializeToString())


class InplaceUpdateExecutor(base_executor.BaseExecutor):
  """A Fake noop executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    model = output_dict['output_key'][0]
    model.name = 'my_model'


class PythonExecutorOperatorTest(test_case_utils.TfxTest):

  def testRunExecutor_with_InprocessExecutor(self):
    executor_sepc = text_format.Parse(
        """
      class_path: "tfx.orchestration.portable.python_executor_operator_test.InprocessExecutor"
    """, executable_spec_pb2.PythonClassExecutableSpec())
    operator = python_executor_operator.PythonExecutorOperator(executor_sepc)
    input_dict = {'input_key': [standard_artifacts.Examples()]}
    output_dict = {'output_key': [standard_artifacts.Model()]}
    exec_properties = {'key': 'value'}
    stateful_working_dir = os.path.join(self.tmp_dir, 'stateful_working_dir')
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    executor_output = operator.run_executor(
        data_types.ExecutionInfo(
            execution_id=1,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
            stateful_working_dir=stateful_working_dir,
            execution_output_uri=executor_output_uri))
    self.assertProtoPartiallyEquals(
        """
          output_artifacts {
            key: "output_key"
            value {
              artifacts {
              }
            }
          }""", executor_output)

  def testRunExecutor_with_NotInprocessExecutor(self):
    executor_sepc = text_format.Parse(
        """
      class_path: "tfx.orchestration.portable.python_executor_operator_test.NotInprocessExecutor"
    """, executable_spec_pb2.PythonClassExecutableSpec())
    operator = python_executor_operator.PythonExecutorOperator(executor_sepc)
    input_dict = {'input_key': [standard_artifacts.Examples()]}
    output_dict = {'output_key': [standard_artifacts.Model()]}
    exec_properties = {'key': 'value'}
    stateful_working_dir = os.path.join(self.tmp_dir, 'stateful_working_dir')
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    executor_output = operator.run_executor(
        data_types.ExecutionInfo(
            execution_id=1,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
            stateful_working_dir=stateful_working_dir,
            execution_output_uri=executor_output_uri))
    self.assertProtoPartiallyEquals(
        """
          output_artifacts {
            key: "output_key"
            value {
              artifacts {
              }
            }
          }""", executor_output)

  def testRunExecutor_with_InplaceUpdateExecutor(self):
    executor_sepc = text_format.Parse(
        """
      class_path: "tfx.orchestration.portable.python_executor_operator_test.InplaceUpdateExecutor"
    """, executable_spec_pb2.PythonClassExecutableSpec())
    operator = python_executor_operator.PythonExecutorOperator(executor_sepc)
    input_dict = {'input_key': [standard_artifacts.Examples()]}
    output_dict = {'output_key': [standard_artifacts.Model()]}
    exec_properties = {
        'string': 'value',
        'int': 1,
        'float': 0.0,
        # This should not happen on production and will be
        # dropped.
        'proto': execution_result_pb2.ExecutorOutput()
    }
    stateful_working_dir = os.path.join(self.tmp_dir, 'stateful_working_dir')
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    executor_output = operator.run_executor(
        data_types.ExecutionInfo(
            execution_id=1,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
            stateful_working_dir=stateful_working_dir,
            execution_output_uri=executor_output_uri))
    self.assertProtoPartiallyEquals(
        """
          output_artifacts {
            key: "output_key"
            value {
              artifacts {
                custom_properties {
                  key: "name"
                  value {
                    string_value: "my_model"
                  }
                }
              }
            }
          }""", executor_output)


if __name__ == '__main__':
  tf.test.main()
