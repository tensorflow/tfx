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
import unittest

import tensorflow as tf

from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import test_utils
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


class _TestMixins:
  """Helper methods for inputs_utils_test."""

  def get_metadata(self):
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    connection_config.sqlite.filename_uri = os.path.join(
        self.get_temp_dir(), 'metadata.db')
    return metadata.Metadata(connection_config)

  def load_pipeline_proto(self, filename):
    result = pipeline_pb2.Pipeline()
    self.load_proto_from_text(os.path.join(_TESTDATA_DIR, filename), result)
    return result

  def _make_artifact(self, type_name, **kwargs):
    result = types.Artifact(
        metadata_store_pb2.ArtifactType(name=type_name))
    for key, value in kwargs.items():
      setattr(result, key, value)
    return result

  def make_examples(self, **kwargs) -> types.Artifact:
    return self._make_artifact('Examples', **kwargs)

  def make_model(self, **kwargs) -> types.Artifact:
    return self._make_artifact('Model', **kwargs)

  def fake_execute(self, metadata_handler, pipeline_node, input_map,
                   output_map):
    contexts = context_lib.register_contexts_if_not_exists(
        metadata_handler, pipeline_node.contexts)
    execution = execution_publish_utils.register_execution(
        metadata_handler, pipeline_node.node_info.type, contexts, input_map)
    execution_publish_utils.publish_succeeded_execution(
        metadata_handler, execution.id, contexts, output_map)

  def assertArtifactEqual(self, expected, actual):
    self.assertProtoPartiallyEquals(
        expected.mlmd_artifact,
        actual.mlmd_artifact,
        ignored_fields=[
            'create_time_since_epoch',
            'last_update_time_since_epoch',
        ])


class InputsUtilsTest(test_utils.TfxTest, _TestMixins):

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

  def testResolveInputsArtifacts(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    another_example_gen = pipeline.nodes[1].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node
    my_trainer = pipeline.nodes[3].pipeline_node

    with self.get_metadata() as m:
      # Publishes first ExampleGen with two output channels. `output_examples`
      # will be consumed by downstream Transform.
      output_example = self.make_examples(uri='my_examples_uri')
      side_examples = self.make_examples(uri='side_examples_uri')
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={
                            'output_examples': [output_example],
                            'another_examples': [side_examples]
                        })

      # Publishes second ExampleGen with one output channel with the same output
      # key as the first ExampleGen. However this is not consumed by downstream
      # nodes.
      another_output_example = self.make_examples(uri='another_examples_uri')
      self.fake_execute(m, another_example_gen,
                        input_map=None,
                        output_map={
                            'output_examples': [another_output_example]
                        })

      # Gets inputs for transform. Should get back what the first ExampleGen
      # published in the `output_examples` channel.
      transform_inputs = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_transform.inputs)
      self.assertEqual(len(transform_inputs), 1)
      self.assertEqual(len(transform_inputs['examples']), 1)
      self.assertArtifactEqual(transform_inputs['examples'][0], output_example)

      # Tries to resolve inputs for trainer. As trainer also requires min_count
      # for both input channels (from example_gen and from transform) but we did
      # not publish anything from transform, it should return nothing.
      self.assertIsNone(
          inputs_utils.resolve_input_artifacts(
              context=base_resolver.ResolverContext(metadata_handler=m),
              node_inputs=my_trainer.inputs))

  def testResolverWithLatestArtifactsResolver(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    with self.get_metadata() as m:
      # Publishes first ExampleGen with two output channels. `output_examples`
      # will be consumed by downstream Transform.
      output_example_1 = self.make_examples(uri='my_examples_uri_1')
      output_example_2 = self.make_examples(uri='my_examples_uri_2')
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={
                            'output_examples': [output_example_1,
                                                output_example_2]
                        })

      transform_resolver = my_transform.inputs.resolver_config.resolvers.add()
      transform_resolver.name = 'LatestArtifactsResolver'
      transform_resolver.config_json = '{}'

      # Gets inputs for transform. Should get back what the first ExampleGen
      # published in the `output_examples` channel.
      transform_inputs = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_transform.inputs)
      self.assertEqual(len(transform_inputs), 1)
      self.assertEqual(len(transform_inputs['examples']), 1)
      self.assertArtifactEqual(transform_inputs['examples'][0],
                               output_example_2)

  def testResolveInputArtifactsOutputKeyUnset(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test_output_key_unset.pbtxt')
    my_trainer = pipeline.nodes[0].pipeline_node
    my_pusher = pipeline.nodes[1].pipeline_node

    with self.get_metadata() as m:
      # Publishes Trainer with one output channels. `output_model`
      # will be consumed by the Pusher in the different run.
      output_model = self.make_model(uri='my_output_model_uri')
      self.fake_execute(m, my_trainer,
                        input_map=None,
                        output_map={
                            'model': [output_model]
                        })

      # Gets inputs for pusher. Should get back what the first Model
      # published in the `output_model` channel.
      pusher_inputs = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_pusher.inputs)
      self.assertEqual(len(pusher_inputs), 1)
      self.assertEqual(len(pusher_inputs['model']), 1)
      self.assertArtifactEqual(output_model, pusher_inputs['model'][0])


@unittest.skip('UnprocessedArtifactsResolver not available.')
class InputsUtilsResolverTests(test_utils.TfxTest, _TestMixins):

  def setUp(self):
    super().setUp()
    with self.get_metadata() as m:
      common_utils.register_type_if_not_exist(
          m, metadata_store_pb2.ExecutionType(name='Transform'))

  def testLatestUnprocessedArtifacts(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolvers.add()
    resolver1.name = 'UnprocessedArtifactsResolver'
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolvers.add()
    resolver2.name = 'LatestArtifactsResolver'
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex1]})
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex2]})

      result = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_transform.inputs)

    self.assertIsNotNone(result)
    self.assertLen(result['examples'], 1)
    self.assertArtifactEqual(result['examples'][0], ex2)

  def testLatestUnprocessedArtifacts_IgnoreAlreadyProcessed(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolvers.add()
    resolver1.name = 'UnprocessedArtifactsResolver'
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolvers.add()
    resolver2.name = 'LatestArtifactsResolver'
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex1]})
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex2]})
      self.fake_execute(m, my_transform,
                        input_map={'examples': [ex2]},
                        output_map=None)

      result = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_transform.inputs)

    self.assertIsNotNone(result)
    self.assertLen(result['examples'], 1)
    self.assertArtifactEqual(result['examples'][0], ex1)

  def testLatestUnprocessedArtifacts_NoneIfEverythingProcessed(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolvers.add()
    resolver1.name = 'UnprocessedArtifactsResolver'
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolvers.add()
    resolver2.name = 'LatestArtifactsResolver'
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex1]})
      self.fake_execute(m, my_example_gen,
                        input_map=None,
                        output_map={'output_examples': [ex2]})
      self.fake_execute(m, my_transform,
                        input_map={'examples': [ex1]},
                        output_map=None)
      self.fake_execute(m, my_transform,
                        input_map={'examples': [ex2]},
                        output_map=None)

      result = inputs_utils.resolve_input_artifacts(
          context=base_resolver.ResolverContext(metadata_handler=m),
          node_inputs=my_transform.inputs)

    self.assertIsNone(result)


if __name__ == '__main__':
  tf.test.main()
