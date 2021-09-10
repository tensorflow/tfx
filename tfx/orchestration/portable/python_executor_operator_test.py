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
from typing import Any, Dict, List

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import python_executor_operator
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format


class InprocessExecutor(base_executor.BaseExecutor):
  """A Fake in-process executor what returns execution result."""

  def Do(
      self, input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any]) -> execution_result_pb2.ExecutorOutput:
    executor_output = execution_result_pb2.ExecutorOutput()
    outputs_utils.populate_output_artifact(executor_output, output_dict)
    return executor_output


class NotInprocessExecutor(base_executor.BaseExecutor):
  """A Fake not-in-process executor what writes execution result to executor_output_uri."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    executor_output = execution_result_pb2.ExecutorOutput()
    outputs_utils.populate_output_artifact(executor_output, output_dict)
    with fileio.open(self._context.executor_output_uri, 'wb') as f:
      f.write(executor_output.SerializeToString())


class InplaceUpdateExecutor(base_executor.BaseExecutor):
  """A Fake executor that uses the executor Context to compute its output."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    model = output_dict['output_key'][0]
    model.name = '{0}.{1}.my_model'.format(
        self._context.pipeline_info.id,
        self._context.pipeline_node.node_info.id)


class PythonExecutorOperatorTest(test_case_utils.TfxTest):

  def _get_execution_info(self, input_dict, output_dict, exec_properties):
    pipeline_node = pipeline_pb2.PipelineNode(node_info={'id': 'MyPythonNode'})
    pipeline_info = pipeline_pb2.PipelineInfo(id='MyPipeline')
    stateful_working_dir = os.path.join(self.tmp_dir, 'stateful_working_dir')
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    return data_types.ExecutionInfo(
        execution_id=1,
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        stateful_working_dir=stateful_working_dir,
        execution_output_uri=executor_output_uri,
        pipeline_node=pipeline_node,
        pipeline_info=pipeline_info,
        pipeline_run_id=99)

  def testRunExecutor_with_InprocessExecutor(self):
    executor_sepc = text_format.Parse(
        """
      class_path: "tfx.orchestration.portable.python_executor_operator_test.InprocessExecutor"
    """, executable_spec_pb2.PythonClassExecutableSpec())
    operator = python_executor_operator.PythonExecutorOperator(executor_sepc)
    input_dict = {'input_key': [standard_artifacts.Examples()]}
    output_dict = {'output_key': [standard_artifacts.Model()]}
    exec_properties = {'key': 'value'}
    executor_output = operator.run_executor(
        self._get_execution_info(input_dict, output_dict, exec_properties))
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
    executor_output = operator.run_executor(
        self._get_execution_info(input_dict, output_dict, exec_properties))
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
    executor_output = operator.run_executor(
        self._get_execution_info(input_dict, output_dict, exec_properties))
    self.assertProtoPartiallyEquals(
        """
          output_artifacts {
            key: "output_key"
            value {
              artifacts {
                custom_properties {
                  key: "name"
                  value {
                    string_value: "MyPipeline.MyPythonNode.my_model"
                  }
                }
              }
            }
          }""", executor_output)


if __name__ == '__main__':
  tf.test.main()
