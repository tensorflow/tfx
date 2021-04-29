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
"""Tests for tfx.orchestration.kubernetes.kubernetes_dag_runner."""

from unittest import mock
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.types.component_spec import ChannelParameter

from ml_metadata.proto import metadata_store_pb2

_executed_components = []


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


def _initialize_executed_components():
  global _executed_components
  _executed_components = []


def _mock_launch_container_component(component: base_node.BaseNode, *_):
  _executed_components.append(component.id)


# We define fake component spec classes below for testing. Note that we can't
# programmatically generate component using anonymous classes for testing
# because of a limitation in the "dill" pickler component used by Apache Beam.
# An alternative we considered but rejected here was to write a function that
# returns anonymous classes within that function's closure (as is done in
# tfx/orchestration/pipeline_test.py), but that strategy does not work here
# as these anonymous classes cannot be used with Beam, since they cannot be
# pickled with the "dill" library.
class _FakeComponentSpecA(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeA)}


class _FakeComponentSpecB(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeB)}


class _FakeComponentSpecC(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeC)}


class _FakeComponentSpecD(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'b': ChannelParameter(type=_ArtifactTypeB),
      'c': ChannelParameter(type=_ArtifactTypeC),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeD)}


class _FakeComponentSpecE(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'a': ChannelParameter(type=_ArtifactTypeA),
      'b': ChannelParameter(type=_ArtifactTypeB),
      'd': ChannelParameter(type=_ArtifactTypeD),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeE)}


class _FakeComponentSpecF(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'a': ChannelParameter(type=_ArtifactTypeA),
  }
  OUTPUTS = {}


class _FakeComponent(base_component.BaseComponent):

  SPEC_CLASS = types.ComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, spec: types.ComponentSpec):
    super(_FakeComponent, self).__init__(spec=spec)
    self._id = spec.__class__.__name__.replace('_FakeComponentSpec', '').lower()


class KubernetesDagRunnerTest(tf.test.TestCase):

  @mock.patch.object(
      kubernetes_dag_runner,
      'launch_container_component',
      _mock_launch_container_component,
  )
  @mock.patch.object(kubernetes_dag_runner, 'kube_utils')
  def testRun(self, mock_kube_utils):
    _initialize_executed_components()
    mock_kube_utils.is_inside_cluster.return_value = True

    component_a = _FakeComponent(
        spec=_FakeComponentSpecA(output=types.Channel(type=_ArtifactTypeA)))
    component_b = _FakeComponent(
        spec=_FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeB)))
    component_c = _FakeComponent(
        spec=_FakeComponentSpecC(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeC)))
    component_c.add_upstream_node(component_b)
    component_d = _FakeComponent(
        spec=_FakeComponentSpecD(
            b=component_b.outputs['output'],
            c=component_c.outputs['output'],
            output=types.Channel(type=_ArtifactTypeD)))
    component_e = _FakeComponent(
        spec=_FakeComponentSpecE(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            d=component_d.outputs['output'],
            output=types.Channel(type=_ArtifactTypeE)))

    test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[
            component_d, component_c, component_a, component_b, component_e
        ])

    kubernetes_dag_runner.KubernetesDagRunner().run(test_pipeline)
    self.assertEqual(
        _executed_components,
        ['a.Wrapper', 'b.Wrapper', 'c.Wrapper', 'd.Wrapper', 'e.Wrapper'])

  @mock.patch.object(
      kubernetes_dag_runner,
      'launch_container_component',
      _mock_launch_container_component,
  )
  @mock.patch.object(kubernetes_dag_runner, 'kube_utils')
  def testRunWithSameSpec(self, mock_kube_utils):
    _initialize_executed_components()
    mock_kube_utils.is_inside_cluster.return_value = True

    component_a = _FakeComponent(
        spec=_FakeComponentSpecA(output=types.Channel(type=_ArtifactTypeA)))
    component_f1 = _FakeComponent(
        spec=_FakeComponentSpecF(a=component_a.outputs['output'])).with_id('f1')
    component_f2 = _FakeComponent(
        spec=_FakeComponentSpecF(a=component_a.outputs['output'])).with_id('f2')
    component_f2.add_upstream_node(component_f1)

    test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[component_f1, component_f2, component_a])
    kubernetes_dag_runner.KubernetesDagRunner().run(test_pipeline)
    self.assertEqual(_executed_components,
                     ['a.Wrapper', 'f1.Wrapper', 'f2.Wrapper'])


if __name__ == '__main__':
  tf.test.main()
