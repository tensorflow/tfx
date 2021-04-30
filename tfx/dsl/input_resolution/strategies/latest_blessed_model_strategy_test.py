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
"""Test for LatestBlessedModelStrategy."""

import tensorflow as tf
from tfx import types
from tfx.components.model_validator import constants as model_validator
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

_PIPELING_INFO = text_format.Parse("""
  id: "my_pipeline"
""", pipeline_pb2.PipelineInfo())

_PIPELINE_RUNTIME_SPEC = text_format.Parse(
    """
  pipeline_root {
    field_value {
      string_value: "/tmp"
    }
  }
  pipeline_run_id {
    field_value {
      string_value: "my_run_id"
    }
  }
""", pipeline_pb2.PipelineRuntimeSpec())

_PIPLINE_NODE = text_format.Parse(
    """
  node_info {
    id: "test_node"
  }
  inputs {
    inputs {
      key: "model"
      value {
        channels {
          artifact_query {
            type {
              name: "Model"
            }
          }
          producer_node_query {
            id: "my_component"
          }
        }
      }
    }
    inputs {
      key: "model_blessing"
      value {
        channels {
          artifact_query {
            type {
              name: "ModelBlessing"
            }
          }
          producer_node_query {
            id: "my_component"
          }
        }
      }
    }
  }
""", pipeline_pb2.PipelineNode())


class LatestBlessedModelStrategyTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._metadata = self.enter_context(
        metadata.Metadata(connection_config=self._connection_config))
    self._store = self._metadata.store
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def _set_model_blessing_bit(self, artifact: types.Artifact, model_id: int,
                              is_blessed: int):
    artifact.mlmd_artifact.custom_properties[
        model_validator.ARTIFACT_PROPERTY_BLESSED_KEY].int_value = is_blessed
    artifact.mlmd_artifact.custom_properties[
        model_validator
        .ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY].int_value = model_id

  def testStrategy(self):
    contexts = self._metadata.register_pipeline_contexts_if_not_exists(
        self._pipeline_info)
    # Model with id 1, will be blessed.
    model_one = standard_artifacts.Model()
    model_one.uri = 'model_one'
    self._metadata.publish_artifacts([model_one])
    # Model with id 2, will be blessed.
    model_two = standard_artifacts.Model()
    model_two.uri = 'model_two'
    self._metadata.publish_artifacts([model_two])
    # Model with id 3, will not be blessed.
    model_three = standard_artifacts.Model()
    model_three.uri = 'model_three'
    self._metadata.publish_artifacts([model_three])

    model_blessing_one = standard_artifacts.ModelBlessing()
    self._set_model_blessing_bit(model_blessing_one, model_one.id, 1)
    model_blessing_two = standard_artifacts.ModelBlessing()
    self._set_model_blessing_bit(model_blessing_two, model_two.id, 1)
    self._metadata.publish_artifacts([model_blessing_one, model_blessing_two])

    self._metadata.register_execution(
        exec_properties={},
        pipeline_info=self._pipeline_info,
        component_info=self._component_info,
        contexts=contexts)
    self._metadata.publish_execution(
        component_info=self._component_info,
        output_artifacts={
            'a': [model_one, model_two, model_three],
            'b': [model_blessing_one, model_blessing_two]
        })

    strategy = latest_blessed_model_strategy.LatestBlessedModelStrategy()
    resolve_result = strategy.resolve(
        pipeline_info=self._pipeline_info,
        metadata_handler=self._metadata,
        source_channels={
            'model':
                types.Channel(
                    type=standard_artifacts.Model,
                    producer_component_id=self._component_info.component_id,
                    output_key='a'),
            'model_blessing':
                types.Channel(
                    type=standard_artifacts.ModelBlessing,
                    producer_component_id=self._component_info.component_id,
                    output_key='b')
        })
    self.assertTrue(resolve_result.has_complete_result)
    self.assertEqual([
        a.uri
        for a in resolve_result.per_key_resolve_result['model']
    ], ['model_two'])
    self.assertTrue(resolve_result.per_key_resolve_state['model'])

  def testStrategy_IrMode(self):
    # Model with id 1, will be blessed.
    model_one = standard_artifacts.Model()
    model_one.uri = 'model_one'
    model_one.id = 1
    # Model with id 2, will be blessed.
    model_two = standard_artifacts.Model()
    model_two.uri = 'model_two'
    model_two.id = 2
    # Model with id 3, will not be blessed.
    model_three = standard_artifacts.Model()
    model_three.uri = 'model_three'
    model_three.id = 3

    model_blessing_one = standard_artifacts.ModelBlessing()
    self._set_model_blessing_bit(model_blessing_one, model_one.id, 1)
    model_blessing_two = standard_artifacts.ModelBlessing()
    self._set_model_blessing_bit(model_blessing_two, model_two.id, 1)

    strategy = latest_blessed_model_strategy.LatestBlessedModelStrategy()
    result = strategy.resolve_artifacts(
        self._store, {
            'model': [model_one, model_two, model_three],
            'model_blessing': [model_blessing_one, model_blessing_two]
        })
    self.assertIsNotNone(result)
    self.assertEqual([a.uri for a in result['model']], ['model_two'])


if __name__ == '__main__':
  tf.test.main()
