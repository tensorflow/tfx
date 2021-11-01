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
"""Tests for tfx.dsl.components.base.base_beam_component."""

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec

_TestBeamPipelineArgs = ["--my_testing_beam_pipeline_args=foo"]


class _EmptyComponentSpec(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {}


class ComponentTest(tf.test.TestCase):

  def testWithBeamPipelineArgs(self):

    class BeamComponent(base_beam_component.BaseBeamComponent):
      EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
          base_beam_executor.BaseBeamExecutor)
      SPEC_CLASS = _EmptyComponentSpec

    beam_component = BeamComponent(spec=_EmptyComponentSpec(
    )).with_beam_pipeline_args(_TestBeamPipelineArgs)
    self.assertEqual(beam_component.executor_spec.beam_pipeline_args,
                     _TestBeamPipelineArgs)

  def testComponentExecutorClass(self):

    class InvalidExecutorComponent(base_beam_component.BaseBeamComponent):
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)
      SPEC_CLASS = _EmptyComponentSpec

    with self.assertRaisesRegex(
        TypeError, "expects EXECUTOR_SPEC property to be an instance of "
        "BeamExecutorSpec"):
      InvalidExecutorComponent._validate_component_class()

if __name__ == "__main__":
  tf.test.main()
