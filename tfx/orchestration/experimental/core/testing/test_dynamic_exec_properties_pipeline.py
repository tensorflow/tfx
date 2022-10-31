# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Pipeline for testing Dynamic Exec Properties.
"""

import os
from typing import Any, Dict, List, Union, Optional

from tfx import types
from tfx.dsl.compiler import compiler
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import component_spec
from tfx.types import standard_artifacts

_pipeline_name = 'dynamic_exec_properties_pipeline'
_pipeline_root = os.path.join('pipeline', _pipeline_name)


@component
def UpstreamComponent(start_num: Parameter[int],   # pylint: disable=invalid-name
                      num: OutputArtifact[standard_artifacts.Integer]):
  num.value = start_num + 1


class DownstreamSpec(types.ComponentSpec):
  PARAMETERS = {
      'input_num':
          component_spec.ExecutionParameter(type=int)
  }
  INPUTS = {}
  OUTPUTS = {}


class Executor(base_executor.BaseExecutor):
  """Executor for test component.
  """

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    assert exec_properties['input_num']
    return


class DownstreamComponent(base_component.BaseComponent):
  """DownstreamComponent is an experimental component.

  Component parameters include a dynamic execution prop to take upstream output.
  """
  SPEC_CLASS = DownstreamSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

  def __init__(self,
               input_num: Optional[Union[int, ph.Placeholder]] = None):
    spec = DownstreamSpec(input_num=input_num)
    super().__init__(spec=spec)


def create_components() -> List[base_component.BaseComponent]:
  upstream_component = UpstreamComponent(start_num=1)  # pylint: disable=no-value-for-parameter
  downstream_component = DownstreamComponent(
      input_num=upstream_component.outputs['num'].future()[0].value)
  return [upstream_component, downstream_component]


def create_pipeline() -> pipeline_pb2.Pipeline:  # pylint: disable=invalid-name
  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=create_components())
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
