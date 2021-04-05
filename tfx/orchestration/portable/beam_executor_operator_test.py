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
from typing import Any, Dict, List, Text

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_executor
from tfx.orchestration.portable import beam_executor_operator
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import executable_spec_pb2
from tfx.utils import test_case_utils

from google.protobuf import text_format


class ValidateBeamPipelineArgsExecutor(base_beam_executor.BaseBeamExecutor):
  """A Fake executor for validating beam pipeline args passing."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    assert '--runner=DirectRunner' in self._beam_pipeline_args


class BeamExecutorOperatorTest(test_case_utils.TfxTest):

  def testRunExecutorWithBeamPipelineArgs(self):
    executor_spec = text_format.Parse(
        """
      python_executor_spec: {
          class_path: "tfx.orchestration.portable.beam_executor_operator_test.ValidateBeamPipelineArgsExecutor"
      }
      beam_pipeline_args: "--runner=DirectRunner"
    """, executable_spec_pb2.BeamExecutableSpec())
    operator = beam_executor_operator.BeamExecutorOperator(executor_spec)
    executor_output_uri = os.path.join(self.tmp_dir, 'executor_output')
    operator.run_executor(
        data_types.ExecutionInfo(
            execution_id=1,
            input_dict={},
            output_dict={},
            exec_properties={},
            execution_output_uri=executor_output_uri))


if __name__ == '__main__':
  tf.test.main()
