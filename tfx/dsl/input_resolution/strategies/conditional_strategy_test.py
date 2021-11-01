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
"""Tests for tfx.dsl.input_resolution.strategies.conditional_strategy."""

import tensorflow as tf
from tfx.dsl.input_resolution.strategies import conditional_strategy
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

_TEST_PREDICATE_1 = """
  operator {
    compare_op {
      lhs {
        operator {
          artifact_value_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      key: "channel_1_key"
                    }
                  }
                }
              }
            }
          }
        }
      }
      rhs {
        value {
          int_value: 0
        }
      }
      op: EQUAL
    }
  }
"""

_TEST_PREDICATE_2 = """
  operator {
    compare_op {
      lhs {
        operator {
          artifact_value_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      key: "channel_2_key"
                    }
                  }
                }
              }
            }
          }
        }
      }
      rhs {
        value {
          int_value: 1
        }
      }
      op: EQUAL
    }
  }
"""


class ConditionalStrategyTest(test_case_utils.TfxTest):

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

  def testStrategy_IrMode_PredicateTrue(self):
    artifact_1 = standard_artifacts.Integer()
    artifact_1.uri = self.create_tempfile().full_path
    artifact_1.value = 0
    artifact_2 = standard_artifacts.Integer()
    artifact_2.uri = self.create_tempfile().full_path
    artifact_2.value = 1

    strategy = conditional_strategy.ConditionalStrategy([
        text_format.Parse(_TEST_PREDICATE_1,
                          placeholder_pb2.PlaceholderExpression()),
        text_format.Parse(_TEST_PREDICATE_2,
                          placeholder_pb2.PlaceholderExpression())
    ])
    input_dict = {'channel_1_key': [artifact_1], 'channel_2_key': [artifact_2]}
    result = strategy.resolve_artifacts(self._store, input_dict)
    self.assertIsNotNone(result)
    self.assertEqual(result, input_dict)

  def testStrategy_IrMode_PredicateFalse(self):
    artifact_1 = standard_artifacts.Integer()
    artifact_1.uri = self.create_tempfile().full_path
    artifact_1.value = 0
    artifact_2 = standard_artifacts.Integer()
    artifact_2.uri = self.create_tempfile().full_path
    artifact_2.value = 42

    strategy = conditional_strategy.ConditionalStrategy([
        text_format.Parse(_TEST_PREDICATE_1,
                          placeholder_pb2.PlaceholderExpression()),
        text_format.Parse(_TEST_PREDICATE_2,
                          placeholder_pb2.PlaceholderExpression())
    ])
    input_dict = {'channel_1_key': [artifact_1], 'channel_2_key': [artifact_2]}
    with self.assertRaises(exceptions.SkipSignal):
      strategy.resolve_artifacts(self._store, input_dict)

if __name__ == '__main__':
  tf.test.main()
