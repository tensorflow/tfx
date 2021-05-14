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
import importlib
import os
import unittest

import tensorflow as tf

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


class _TestMixin:
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

  def make_transform_graph(self, **kwargs) -> types.Artifact:
    return self._make_artifact('TransformGraph', **kwargs)

  def make_model(self, **kwargs) -> types.Artifact:
    return self._make_artifact('Model', **kwargs)

  def fake_execute(self, metadata_handler, pipeline_node, input_map,
                   output_map):
    contexts = context_lib.prepare_contexts(
        metadata_handler, pipeline_node.contexts)
    execution = execution_publish_utils.register_execution(
        metadata_handler, pipeline_node.node_info.type, contexts, input_map)
    return execution_publish_utils.publish_succeeded_execution(
        metadata_handler, execution.id, contexts, output_map)

  def assertArtifactEqual(self, expected, actual):
    self.assertProtoPartiallyEquals(
        expected.mlmd_artifact,
        actual.mlmd_artifact,
        ignored_fields=[
            'create_time_since_epoch',
            'last_update_time_since_epoch',
        ])

  def assertArtifactMapEqual(self, expected, actual):
    self.assertIsInstance(expected, dict)
    self.assertIsInstance(actual, dict)
    self.assertEqual(set(expected.keys()), set(actual.keys()))
    for input_key in expected:
      self.assertIsInstance(expected[input_key], list)
      self.assertIsInstance(actual[input_key], list)
      self.assertEqual(len(expected[input_key]), len(actual[input_key]))
      for expected_item, actual_item in zip(expected[input_key],
                                            actual[input_key]):
        self.assertIsInstance(expected_item, types.Artifact)
        self.assertIsInstance(actual_item, types.Artifact)
        self.assertArtifactEqual(expected_item, actual_item)


class InputsUtilsTest(test_case_utils.TfxTest, _TestMixin):

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
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={
              'output_examples': [output_example],
              'another_examples': [side_examples]
          })
      output_example = output_artifacts['output_examples'][0]
      side_examples = output_artifacts['another_examples'][0]

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
          m, my_transform.inputs)
      self.assertArtifactMapEqual({'examples': [output_example]},
                                  transform_inputs)

      # Tries to resolve inputs for trainer. As trainer also requires min_count
      # for both input channels (from example_gen and from transform) but we did
      # not publish anything from transform, it should return nothing.
      self.assertIsNone(
          inputs_utils.resolve_input_artifacts(m, my_trainer.inputs))

  def testResolverWithLatestArtifactStrategy(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    with self.get_metadata() as m:
      # Publishes first ExampleGen with two output channels. `output_examples`
      # will be consumed by downstream Transform.
      output_example_1 = self.make_examples(uri='my_examples_uri_1')
      output_example_2 = self.make_examples(uri='my_examples_uri_2')
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [output_example_1, output_example_2]})
      output_example_1 = output_artifacts['output_examples'][0]
      output_example_2 = output_artifacts['output_examples'][1]

      transform_resolver = (my_transform.inputs.resolver_config
                            .resolver_steps.add())
      transform_resolver.class_path = (
          'tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
          '.LatestArtifactStrategy')
      transform_resolver.config_json = '{}'

      # Gets inputs for transform. Should get back what the first ExampleGen
      # published in the `output_examples` channel.
      transform_inputs = inputs_utils.resolve_input_artifacts(
          m, my_transform.inputs)
      self.assertArtifactMapEqual({'examples': [output_example_2]},
                                  transform_inputs)

  def testResolveInputArtifactsOutputKeyUnset(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test_output_key_unset.pbtxt')
    my_trainer = pipeline.nodes[0].pipeline_node
    my_pusher = pipeline.nodes[1].pipeline_node

    with self.get_metadata() as m:
      # Publishes Trainer with one output channels. `output_model`
      # will be consumed by the Pusher in the different run.
      output_model = self.make_model(uri='my_output_model_uri')
      output_artifacts = self.fake_execute(
          m, my_trainer, input_map=None, output_map={'model': [output_model]})
      output_model = output_artifacts['model'][0]

      # Gets inputs for pusher. Should get back what the first Model
      # published in the `output_model` channel.
      pusher_inputs = inputs_utils.resolve_input_artifacts(
          m, my_pusher.inputs)
      self.assertArtifactMapEqual({'model': [output_model]},
                                  pusher_inputs)


def unprocessed_artifacts_resolvers_available():
  try:
    importlib.import_module(
        'tfx.dsl.resolvers.unprocessed_artifacts_resolver')
  except ImportError:
    return False
  else:
    return True


class InputsUtilsResolverTests(test_case_utils.TfxTest, _TestMixin):

  def setUp(self):
    super().setUp()
    with self.get_metadata() as m:
      common_utils.register_type_if_not_exist(
          m, metadata_store_pb2.ExecutionType(name='Transform'))
      common_utils.register_type_if_not_exist(
          m, metadata_store_pb2.ExecutionType(name='Trainer'))

  @unittest.skipUnless(unprocessed_artifacts_resolvers_available(),
                       'UnprocessedArtifactsResolver not available.')
  def testLatestUnprocessedArtifacts(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver1.class_path = (
        'tfx.dsl.resolvers.unprocessed_artifacts_resolver'
        '.UnprocessedArtifactsResolver')
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver2.class_path = (
        'tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
        '.LatestArtifactStrategy')
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex1]})
      ex1 = output_artifacts['output_examples'][0]
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex2]})
      ex2 = output_artifacts['output_examples'][0]

      result = inputs_utils.resolve_input_artifacts(
          metadata_handler=m,
          node_inputs=my_transform.inputs)

    self.assertArtifactMapEqual({'examples': [ex2]}, result)

  @unittest.skipUnless(unprocessed_artifacts_resolvers_available(),
                       'UnprocessedArtifactsResolver not available.')
  def testLatestUnprocessedArtifacts_IgnoreAlreadyProcessed(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver1.class_path = (
        'tfx.dsl.resolvers.unprocessed_artifacts_resolver'
        '.UnprocessedArtifactsResolver')
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver2.class_path = (
        'tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
        '.LatestArtifactStrategy')
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex1]})
      ex1 = output_artifacts['output_examples'][0]
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex2]})
      ex2 = output_artifacts['output_examples'][0]
      self.fake_execute(
          m, my_transform, input_map={'examples': [ex2]}, output_map=None)

      result = inputs_utils.resolve_input_artifacts(
          metadata_handler=m,
          node_inputs=my_transform.inputs)

    self.assertArtifactMapEqual({'examples': [ex1]}, result)

  @unittest.skipUnless(unprocessed_artifacts_resolvers_available(),
                       'UnprocessedArtifactsResolver not available.')
  def testLatestUnprocessedArtifacts_NoneIfEverythingProcessed(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node

    resolver1 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver1.class_path = (
        'tfx.dsl.resolvers.unprocessed_artifacts_resolver'
        '.UnprocessedArtifactsResolver')
    resolver1.config_json = '{"execution_type_name": "Transform"}'
    resolver2 = my_transform.inputs.resolver_config.resolver_steps.add()
    resolver2.class_path = (
        'tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
        '.LatestArtifactStrategy')
    resolver2.config_json = '{}'

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='a')
      ex2 = self.make_examples(uri='b')
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex1]})
      ex1 = output_artifacts['output_examples'][0]
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex2]})
      ex2 = output_artifacts['output_examples'][0]
      self.fake_execute(m, my_transform,
                        input_map={'examples': [ex1]},
                        output_map=None)
      self.fake_execute(m, my_transform,
                        input_map={'examples': [ex2]},
                        output_map=None)

      result = inputs_utils.resolve_input_artifacts(
          metadata_handler=m,
          node_inputs=my_transform.inputs)

    self.assertIsNone(result)

  def testLatestArtifacts_withInputKeys(self):
    pipeline = self.load_pipeline_proto(
        'pipeline_for_input_resolver_test.pbtxt')
    my_example_gen = pipeline.nodes[0].pipeline_node
    my_transform = pipeline.nodes[2].pipeline_node
    my_trainer = pipeline.nodes[3].pipeline_node

    # Use LatestArtifactStrategy for TransformGraph only.
    resolver = my_trainer.inputs.resolver_config.resolver_steps.add()
    resolver.class_path = (
        'tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
        '.LatestArtifactStrategy')
    resolver.config_json = '{}'
    resolver.input_keys.append('transform_graph')

    with self.get_metadata() as m:
      ex1 = self.make_examples(uri='examples/1')
      ex2 = self.make_examples(uri='examples/2')
      tf1 = self.make_transform_graph(uri='transform_graph/1')
      tf2 = self.make_transform_graph(uri='transform_graph/2')
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex1]})
      ex1 = output_artifacts['output_examples'][0]
      output_artifacts = self.fake_execute(
          m,
          my_example_gen,
          input_map=None,
          output_map={'output_examples': [ex2]})
      ex2 = output_artifacts['output_examples'][0]
      output_artifacts = self.fake_execute(
          m,
          my_transform,
          input_map={'examples': [ex1]},
          output_map={'transform_graph': [tf1]})
      tf1 = output_artifacts['transform_graph'][0]
      output_artifacts = self.fake_execute(
          m,
          my_transform,
          input_map={'examples': [ex2]},
          output_map={'transform_graph': [tf2]})
      tf2 = output_artifacts['transform_graph'][0]
      result = inputs_utils.resolve_input_artifacts(
          metadata_handler=m,
          node_inputs=my_trainer.inputs)

    # "examples" input channel doesn't go through the resolver and its order is
    # undeterministic. Sort artifacts by ID for testing convenience.
    result['examples'].sort(key=lambda a: a.id)

    self.assertArtifactMapEqual(
        {'examples': [ex1, ex2],
         'transform_graph': [tf2]},
        result)


if __name__ == '__main__':
  tf.test.main()
