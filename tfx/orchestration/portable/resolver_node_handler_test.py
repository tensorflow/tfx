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
"""Tests for tfx.orchestration.portable.resolver_node_handler."""

import os
from unittest import mock

import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import resolver_node_handler
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from google.protobuf import text_format


class ResolverNodeHandlerTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)
    self._testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')

    # Sets up pipelines
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata',
            'pipeline_for_resolver_test.pbtxt'), pipeline)
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'my_pipeline_run',
        })

    # Extracts components
    self._my_trainer = pipeline.nodes[0].pipeline_node
    self._my_resolver = pipeline.nodes[1].pipeline_node
    self._model_type = (
        self._my_trainer.outputs.outputs['model'].artifact_spec.type)

  def _create_model_artifact(self, uri: str) -> types.Artifact:
    result = types.Artifact(self._model_type)
    result.uri = uri
    return result

  def testRun_ExecutionCompleted(self):
    with self._mlmd_connection as m:
      # Publishes two models which will be consumed by downstream resolver.
      output_model_1 = self._create_model_artifact(uri='my_model_uri_1')
      output_model_2 = self._create_model_artifact(uri='my_model_uri_2')

      contexts = context_lib.prepare_contexts(m, self._my_trainer.contexts)
      execution = execution_publish_utils.register_execution(
          m, self._my_trainer.node_info.type, contexts)
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'model': [output_model_1, output_model_2],
          })

    handler = resolver_node_handler.ResolverNodeHandler()
    execution_info = handler.run(
        mlmd_connection=self._mlmd_connection,
        pipeline_node=self._my_resolver,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)

    with self._mlmd_connection as m:
      # There is no way to directly verify the output artifact of the resolver
      # So here a fake downstream component is created which listens to the
      # resolver's output and we verify its input.
      down_stream_node = text_format.Parse(
          """
        inputs {
          inputs {
            key: "input_models"
            value {
              channels {
                producer_node_query {
                  id: "my_resolver"
                }
                context_queries {
                  type {
                    name: "pipeline"
                  }
                  name {
                    field_value {
                      string_value: "my_pipeline"
                    }
                  }
                }
                context_queries {
                  type {
                    name: "component"
                  }
                  name {
                    field_value {
                      string_value: "my_resolver"
                    }
                  }
                }
                artifact_query {
                  type {
                    name: "Model"
                  }
                }
                output_key: "models"
              }
              min_count: 1
            }
          }
        }
        upstream_nodes: "my_resolver"
        """, pipeline_pb2.PipelineNode())
      downstream_input_artifacts = inputs_utils.resolve_input_artifacts(
          metadata_handler=m, pipeline_node=down_stream_node)[0]
      downstream_input_model = downstream_input_artifacts['input_models']
      self.assertLen(downstream_input_model, 1)
      self.assertProtoPartiallyEquals(
          """
          id: 2
          uri: "my_model_uri_2"
          state: LIVE""",
          downstream_input_model[0].mlmd_artifact,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])

      self.assertProtoPartiallyEquals(
          """
          id: 2
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])

  @mock.patch.object(inputs_utils, 'resolve_input_artifacts')
  def testRun_InputResolutionError_ExecutionFailed(self, mock_resolve):
    mock_resolve.side_effect = exceptions.InputResolutionError('Meh')
    handler = resolver_node_handler.ResolverNodeHandler()

    execution_info = handler.run(
        mlmd_connection=self._mlmd_connection,
        pipeline_node=self._my_resolver,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)

    with self._mlmd_connection as m:
      self.assertTrue(execution_info.execution_id)
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          """,
          execution,
          ignored_fields=[
              'type_id', 'custom_properties', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])

  @mock.patch.object(inputs_utils, 'resolve_input_artifacts')
  def testRun_MultipleInputs_ExecutionFailed(self, mock_resolve):
    mock_resolve.return_value = inputs_utils.Trigger([
        {'model': [self._create_model_artifact(uri='/tmp/model/1')]},
        {'model': [self._create_model_artifact(uri='/tmp/model/2')]},
    ])
    handler = resolver_node_handler.ResolverNodeHandler()

    execution_info = handler.run(
        mlmd_connection=self._mlmd_connection,
        pipeline_node=self._my_resolver,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)

    with self._mlmd_connection as m:
      self.assertTrue(execution_info.execution_id)
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          """,
          execution,
          ignored_fields=[
              'type_id', 'custom_properties', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])


if __name__ == '__main__':
  tf.test.main()
