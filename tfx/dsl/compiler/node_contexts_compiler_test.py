# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.compiler.node_contexts_compiler."""

import tensorflow as tf
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import node_contexts_compiler
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format

_NODE_ID = 'test_node'
_PIPELINE_NAME = 'test_pipeline'


class NodeContextsCompilerTest(tf.test.TestCase):

  def test_compile_node_contexts(self):
    expected_node_contexts = text_format.Parse(
        """
        contexts {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "test_pipeline"
            }
          }
        }
        contexts {
          type {
            name: "pipeline_run"
          }
          name {
            runtime_parameter {
              name: "pipeline-run-id"
              type: STRING
            }
          }
        }
        contexts {
          type {
            name: "node"
          }
          name {
            field_value {
              string_value: "test_pipeline.test_node"
            }
          }
        }
        """,
        pipeline_pb2.NodeContexts(),
    )
    self.assertProtoEquals(
        expected_node_contexts,
        node_contexts_compiler.compile_node_contexts(
            compiler_context.PipelineContext(pipeline.Pipeline(_PIPELINE_NAME)),
            _NODE_ID,
        ),
    )

  def test_compile_node_contexts_for_subpipeline(self):
    parent_context = compiler_context.PipelineContext(
        pipeline.Pipeline(_PIPELINE_NAME)
    )
    subpipeline_context = compiler_context.PipelineContext(
        pipeline.Pipeline('subpipeline'), parent_context
    )

    expected_node_contexts = text_format.Parse(
        """
        contexts {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "subpipeline"
            }
          }
        }
        contexts {
          type {
            name: "pipeline_run"
          }
          name {
            structural_runtime_parameter {
              parts {
                constant_value: "subpipeline_"
              }
              parts {
                runtime_parameter {
                  name: "pipeline-run-id"
                  type: STRING
                }
              }
            }
          }
        }
        contexts {
          type {
            name: "node"
          }
          name {
            field_value {
              string_value: "test_pipeline.subpipeline"
            }
          }
        }
        contexts {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "test_pipeline"
            }
          }
        }
        contexts {
          type {
            name: "pipeline_run"
          }
          name {
            runtime_parameter {
              name: "pipeline-run-id"
              type: STRING
            }
          }
        }
        contexts {
          type {
            name: "node"
          }
          name {
            field_value {
              string_value: "subpipeline.test_node"
            }
          }
        }
        """,
        pipeline_pb2.NodeContexts(),
    )
    self.assertProtoEquals(
        expected_node_contexts,
        node_contexts_compiler.compile_node_contexts(
            subpipeline_context,
            _NODE_ID,
        ),
    )
