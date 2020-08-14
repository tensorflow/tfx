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

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from google.protobuf import json_format
import json

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.kubernetes import kubernetes_remote_runner
from tfx.types.component_spec import ChannelParameter
from tfx.utils import json_utils


class _ArtifactTypeA(types.Artifact):
  TYPE_NAME = 'ArtifactTypeA'


class _ArtifactTypeB(types.Artifact):
  TYPE_NAME = 'ArtifactTypeB'


class _FakeComponentSpecA(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeA)}


class _FakeComponentSpecB(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeB)}

class _FakeComponent(base_component.BaseComponent):
  SPEC_CLASS = types.ComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, spec: types.ComponentSpec):
    instance_name = spec.__class__.__name__.replace(
        '_FakeComponentSpec', '').lower()
    super(_FakeComponent, self).__init__(spec=spec, instance_name=instance_name)

class KubernetesRemoteRunnerTest(tf.test.TestCase):

  def setUp(self):
    component_a = _FakeComponent(
        _FakeComponentSpecA(output=types.Channel(type=_ArtifactTypeA)))
    component_b = _FakeComponent(
        _FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeB)))
    self.test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
        components=[
            component_a, component_b
        ])

  def testSerialization(self):
    serialized_pipeline = kubernetes_remote_runner._serialize_pipeline( # pylint: disable=protected-access
        self.test_pipeline)

    tfx_pipeline = json.loads(serialized_pipeline)
    components = [
        json_utils.loads(component) for component in tfx_pipeline['components']
    ]
    metadata_connection_config = metadata_store_pb2.ConnectionConfig()
    json_format.Parse(tfx_pipeline['metadata_connection_config'],
                      metadata_connection_config)
    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_name,
                     tfx_pipeline['pipeline_name'])
    self.assertEqual(self.test_pipeline.pipeline_info.pipeline_root,
                     tfx_pipeline['pipeline_root'])
    self.assertEqual(self.test_pipeline.enable_cache,
                     tfx_pipeline['enable_cache'])
    self.assertEqual(self.test_pipeline.beam_pipeline_args,
                     tfx_pipeline['beam_pipeline_args'])
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     metadata_connection_config)
    self.assertListEqual(
        [component.executor_spec.executor_class for component in
         self.test_pipeline.components],
        [component.executor_spec.executor_class for component in components])
    self.assertEqual(self.test_pipeline.metadata_connection_config,
                     metadata_connection_config)

if __name__ == '__main__':
  tf.test.main()
