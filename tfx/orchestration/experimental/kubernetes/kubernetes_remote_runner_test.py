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
"""Kubernetes TFX runner for out-of-cluster orchestration."""

import json

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.experimental.kubernetes import kubernetes_remote_runner
from tfx.types.component_spec import ChannelParameter
from tfx.utils import json_utils

from google.protobuf import json_format
from ml_metadata.proto import metadata_store_pb2


class _ArtifactTypeA(types.Artifact):
  TYPE_NAME = 'ArtifactTypeA'


class _ArtifactTypeB(types.Artifact):
  TYPE_NAME = 'ArtifactTypeB'


class _ArtifactTypeC(types.Artifact):
  TYPE_NAME = 'ArtifactTypeC'


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
  INPUTS = {
      'a': ChannelParameter(type=_ArtifactTypeA),
      'b': ChannelParameter(type=_ArtifactTypeB)
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeC)}


class _FakeComponent(base_component.BaseComponent):
  SPEC_CLASS = types.ComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, spec: types.ComponentSpec):
    super(_FakeComponent, self).__init__(spec=spec)
    self._id = spec.__class__.__name__.replace('_FakeComponentSpec', '').lower()


class KubernetesRemoteRunnerTest(tf.test.TestCase):

  def setUp(self):
    super(KubernetesRemoteRunnerTest, self).setUp()
    self.component_a = _FakeComponent(
        _FakeComponentSpecA(output=types.Channel(type=_ArtifactTypeA)))
    self.component_b = _FakeComponent(
        _FakeComponentSpecB(
            a=self.component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeB)))
    self.component_c = _FakeComponent(
        _FakeComponentSpecC(
            a=self.component_a.outputs['output'],
            b=self.component_b.outputs['output'],
            output=types.Channel(type=_ArtifactTypeC)))
    self.test_pipeline = tfx_pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[self.component_c, self.component_a, self.component_b])

  def testSerialization(self):
    serialized_pipeline = kubernetes_remote_runner._serialize_pipeline(  # pylint: disable=protected-access
        self.test_pipeline)

    pipeline = json.loads(serialized_pipeline)
    components = [
        json_utils.loads(component) for component in pipeline['components']
    ]
    metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    json_format.Parse(pipeline['metadata_connection_config'],
                      metadata_connection_config)
    expected_downstream_ids = {
        'a': ['b', 'c'],
        'b': ['c'],
        'c': [],
    }
    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_name,
                     pipeline['pipeline_name'])
    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_root,
                     pipeline['pipeline_root'])
    self.assertEqual(self.test_pipeline.enable_cache, pipeline['enable_cache'])
    self.assertEqual(self.test_pipeline.beam_pipeline_args,
                     pipeline['beam_pipeline_args'])
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     metadata_connection_config)
    self.assertListEqual([
        component.executor_spec.executor_class
        for component in self.test_pipeline.components
    ], [component.executor_spec.executor_class for component in components])
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     metadata_connection_config)
    # Enforce order of downstream ids for comparison.
    for downstream_ids in pipeline['downstream_ids'].values():
      downstream_ids.sort()
    self.assertEqual(expected_downstream_ids, pipeline['downstream_ids'])

  def testDeserialization(self):
    serialized_pipeline = kubernetes_remote_runner._serialize_pipeline(  # pylint: disable=protected-access
        self.test_pipeline)
    pipeline = kubernetes_remote_runner.deserialize_pipeline(
        serialized_pipeline)

    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_name,
                     pipeline.pipeline_info.pipeline_name)
    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_root,
                     pipeline.pipeline_info.pipeline_root)
    self.assertEqual(self.test_pipeline.enable_cache, pipeline.enable_cache)
    self.assertEqual(self.test_pipeline.beam_pipeline_args,
                     pipeline.beam_pipeline_args)
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     pipeline.metadata_connection_config)
    self.assertListEqual([
        component.executor_spec.executor_class
        for component in self.test_pipeline.components
    ], [
        component.executor_spec.executor_class
        for component in pipeline.components
    ])
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     pipeline.metadata_connection_config)


if __name__ == '__main__':
  tf.test.main()
