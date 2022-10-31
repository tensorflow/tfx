# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.beam_executor_operator."""

import os
from typing import Any, Dict, List

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.orchestration.portable import beam_executor_operator
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format


class ValidateBeamPipelineArgsExecutor(base_beam_executor.BaseBeamExecutor):
  """A Fake executor for validating beam pipeline args passing."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    assert '--runner=DirectRunner' in self._beam_pipeline_args
    model = output_dict['output_key'][0]
    model.name = '{0}.{1}.my_model'.format(
        self._context.pipeline_info.id,
        self._context.pipeline_node.node_info.id)


class BeamExecutorOperatorTest(test_case_utils.TfxTest):

  def testRunExecutorWithBeamPipelineArgs(self):
    executor_spec = text_format.Parse(
        """
      python_executor_spec: {
          class_path: "tfx.orchestration.portable.beam_executor_operator_test.ValidateBeamPipelineArgsExecutor"
      }
      beam_pipeline_args_placeholders {
        value {
          string_value: "--runner=DirectRunner"
        }
      }
    """, executable_spec_pb2.BeamExecutableSpec())
    operator = beam_executor_operator.BeamExecutorOperator(executor_spec)
    pipeline_node = pipeline_pb2.PipelineNode(node_info={'id': 'MyBeamNode'})
    pipeline_info = pipeline_pb2.PipelineInfo(id='MyPipeline')
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    executor_output = operator.run_executor(
        data_types.ExecutionInfo(
            execution_id=1,
            input_dict={'input_key': [standard_artifacts.Examples()]},
            output_dict={'output_key': [standard_artifacts.Model()]},
            exec_properties={},
            execution_output_uri=executor_output_uri,
            pipeline_node=pipeline_node,
            pipeline_info=pipeline_info,
            pipeline_run_id='99'))
    self.assertProtoPartiallyEquals(
        """
          output_artifacts {
            key: "output_key"
            value {
              artifacts {
                custom_properties {
                  key: "name"
                  value {
                    string_value: "MyPipeline.MyBeamNode.my_model"
                  }
                }
                name: "MyPipeline.MyBeamNode.my_model"
              }
            }
          }""", executor_output)


if __name__ == '__main__':
  tf.test.main()
