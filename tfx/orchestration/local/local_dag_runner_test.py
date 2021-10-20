# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.beam.beam_dag_runner."""

import os
import tempfile
from typing import Any, Dict, List

import absl.testing.absltest
from tfx import types
from tfx.dsl.compiler import compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import pipeline as pipeline_py
from tfx.orchestration.local import local_dag_runner
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.proto.orchestration import pipeline_pb2
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter

_executed_components = []
_COMPONENT_NAME = 'component_name'


class _ArtifactTypeA(types.Artifact):
  TYPE_NAME = 'ArtifactTypeA'


class _ArtifactTypeB(types.Artifact):
  TYPE_NAME = 'ArtifactTypeB'


class _ArtifactTypeC(types.Artifact):
  TYPE_NAME = 'ArtifactTypeC'


class _ArtifactTypeD(types.Artifact):
  TYPE_NAME = 'ArtifactTypeD'


class _ArtifactTypeE(types.Artifact):
  TYPE_NAME = 'ArtifactTypeE'


# We define fake component spec classes below for testing.
class _FakeComponentSpecA(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeA)}


class _FakeComponentSpecB(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeB)}


class _FakeComponentSpecC(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeC)}


class _FakeComponentSpecD(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {
      'b': ChannelParameter(type=_ArtifactTypeB),
      'c': ChannelParameter(type=_ArtifactTypeC),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeD)}


class _FakeComponentSpecE(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {
      'a': ChannelParameter(type=_ArtifactTypeA),
      'b': ChannelParameter(type=_ArtifactTypeB),
      'd': ChannelParameter(type=_ArtifactTypeD),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeE)}


class _FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]):
    _executed_components.append(exec_properties[_COMPONENT_NAME])


def _get_fake_component(spec: types.ComponentSpec):
  component_id = spec.__class__.__name__.replace('_FakeComponentSpec',
                                                 '').lower()

  class _FakeComponent(base_component.BaseComponent):
    SPEC_CLASS = types.ComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

  return _FakeComponent(spec=spec).with_id(component_id)


class LocalDagRunnerTest(absl.testing.absltest.TestCase):

  def setUp(self):
    super().setUp()
    _executed_components.clear()

  def _getTestPipeline(self) -> pipeline_py.Pipeline:  # pylint: disable=invalid-name
    component_a = _get_fake_component(
        _FakeComponentSpecA(
            output=types.Channel(type=_ArtifactTypeA),
            component_name='_FakeComponent.a'))
    component_b = _get_fake_component(
        _FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeB),
            component_name='_FakeComponent.b'))
    component_c = _get_fake_component(
        _FakeComponentSpecC(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeC),
            component_name='_FakeComponent.c'))
    component_c.add_upstream_node(component_b)
    component_d = _get_fake_component(
        _FakeComponentSpecD(
            b=component_b.outputs['output'],
            c=component_c.outputs['output'],
            output=types.Channel(type=_ArtifactTypeD),
            component_name='_FakeComponent.d'))
    component_e = _get_fake_component(
        _FakeComponentSpecE(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            d=component_d.outputs['output'],
            output=types.Channel(type=_ArtifactTypeE),
            component_name='_FakeComponent.e'))

    temp_path = tempfile.mkdtemp()
    pipeline_root_path = os.path.join(temp_path, 'pipeline_root')
    metadata_path = os.path.join(temp_path, 'metadata.db')
    return pipeline_py.Pipeline(
        pipeline_name='test_pipeline',
        pipeline_root=pipeline_root_path,
        metadata_connection_config=sqlite_metadata_connection_config(
            metadata_path),
        components=[
            component_d, component_c, component_a, component_b, component_e
        ])

  def _getTestPipelineIR(self) -> pipeline_pb2.Pipeline:  # pylint: disable=invalid-name
    test_pipeline = self._getTestPipeline()
    c = compiler.Compiler()
    return c.compile(test_pipeline)

  def testRun(self):
    local_dag_runner.LocalDagRunner().run(self._getTestPipeline())
    self.assertEqual(_executed_components, [
        '_FakeComponent.a', '_FakeComponent.b', '_FakeComponent.c',
        '_FakeComponent.d', '_FakeComponent.e'
    ])

  def testRunWithIR(self):
    local_dag_runner.LocalDagRunner().run_with_ir(self._getTestPipelineIR())
    self.assertEqual(_executed_components, [
        '_FakeComponent.a', '_FakeComponent.b', '_FakeComponent.c',
        '_FakeComponent.d', '_FakeComponent.e'
    ])


if __name__ == '__main__':
  absl.testing.absltest.main()
