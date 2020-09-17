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
"""Tests for tfx.orchestration.portable.inputs_utils."""
import os
import tensorflow as tf

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import test_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class InputsUtilsTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')

  def testResolveParameters(self):
    parameters = pipeline_pb2.NodeParameters()
    text_format.Parse(
        """
        parameters {
          key: 'key_one'
          value {
            field_value {string_value: 'value_one'}
          }
        }
        parameters {
          key: 'key_two'
          value {
            field_value {int_value: 2}
          }
        }""", parameters)

    parameters = inputs_utils.resolve_parameters(parameters)
    self.assertEqual(len(parameters), 2)
    self.assertEqual(parameters['key_one'], 'value_one')
    self.assertEqual(parameters['key_two'], 2)

  def testResolveParametersFail(self):
    parameters = pipeline_pb2.NodeParameters()
    text_format.Parse(
        """
        parameters {
          key: 'key_one'
          value {
            runtime_parameter {name: 'rp'}
          }
        }""", parameters)
    with self.assertRaisesRegex(RuntimeError, 'Parameter value not ready'):
      inputs_utils.resolve_parameters(parameters)

  def testResolverInputsArtifacts(self):
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(self._testdata_dir,
                     'pipeline_for_input_resolver_test.pbtxt'), pipeline)
    my_example_gen = pipeline.nodes[0].pipeline_node
    another_example_gen = pipeline.nodes[1].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node
    my_trainer = pipeline.nodes[3].pipeline_node

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    with metadata.Metadata(connection_config=connection_config) as m:
      # Publishes first ExampleGen with two output channels. `output_examples`
      # will be consumed by downstream Transform.
      output_example = types.Artifact(
          my_example_gen.outputs.outputs['output_examples'].artifact_spec.type)
      output_example.uri = 'my_examples_uri'
      side_examples = types.Artifact(
          my_example_gen.outputs.outputs['side_examples'].artifact_spec.type)
      side_examples.uri = 'side_examples_uri'
      contexts = context_lib.register_contexts_if_not_exists(
          m, my_example_gen.contexts)
      execution = execution_publish_utils.register_execution(
          m, my_example_gen.node_info.type, contexts)
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'output_examples': [output_example],
              'another_examples': [side_examples]
          })

      # Publishes second ExampleGen with one output channel with the same output
      # key as the first ExampleGen. However this is not consumed by downstream
      # nodes.
      another_output_example = types.Artifact(
          another_example_gen.outputs.outputs['output_examples'].artifact_spec
          .type)
      another_output_example.uri = 'another_examples_uri'
      contexts = context_lib.register_contexts_if_not_exists(
          m, another_example_gen.contexts)
      execution = execution_publish_utils.register_execution(
          m, another_example_gen.node_info.type, contexts)
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'output_examples': [another_output_example],
          })

      # Gets inputs for transform. Should get back what the first ExampleGen
      # published in the `output_examples` channel.
      transform_inputs = inputs_utils.resolve_input_artifacts(
          m, my_transform.inputs)
      self.assertEqual(len(transform_inputs), 1)
      self.assertEqual(len(transform_inputs['examples']), 1)
      self.assertProtoPartiallyEquals(
          transform_inputs['examples'][0].mlmd_artifact,
          output_example.mlmd_artifact,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

      # Tries to resolve inputs for trainer. As trainer also requires min_count
      # for both input channels (from example_gen and from transform) but we did
      # not publish anything from transform, it should return nothing.
      self.assertIsNone(
          inputs_utils.resolve_input_artifacts(m, my_trainer.inputs))


if __name__ == '__main__':
  tf.test.main()
