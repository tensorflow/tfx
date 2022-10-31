# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.input_resolution.channel_resolver."""

import tensorflow as tf
from tfx.orchestration.portable.input_resolution import channel_resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from google.protobuf import text_format


class ChannelResolverTest(test_case_utils.TfxTest, test_case_utils.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()
    # We have to __enter__ Metadata which activates the MetadataStore so that
    # we can use the same fake in-memory MetadataStore instance during the
    # single test.
    self.enter_context(self.mlmd_handle)

  def make_channel_spec(self, channel_spec_str: str):
    return text_format.Parse(channel_spec_str, pipeline_pb2.InputSpec.Channel())

  def testResolveSingleChannel_NoContextQueries_Empty(self):
    ch = self.make_channel_spec("""
      artifact_query {
        type {
          name: "Examples"
        }
      }
      output_key: "examples"
    """)
    resolved = channel_resolver.resolve_single_channel(
        self.mlmd_handle, ch)
    self.assertEmpty(resolved)

  def testResolveSingleChannel_BadContextQuery(self):
    with self.subTest('No type'):
      ch = self.make_channel_spec("""
        context_queries {
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(
            self.mlmd_handle, ch)

    with self.subTest('No type.name'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            id: 123
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(
            self.mlmd_handle, ch)

    with self.subTest('No name'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
        }
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(
            self.mlmd_handle, ch)

    with self.subTest('Non-existential'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "i dont exist"
          }
          name {
            field_value {
              string_value: "i dont exist"
            }
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertEmpty(resolved)

  def testResolveSingleChannel_AllContexts(self):
    p = self.put_context('pipeline', 'my-pipeline')
    r1 = self.put_context('pipeline_run', 'run-001')
    r2 = self.put_context('pipeline_run', 'run-002')
    self.put_context('hahaha', 'i-am-a-troll')
    e1 = self.put_artifact('Examples')
    e2 = self.put_artifact('Examples')
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [e1]},
        contexts=[p, r1])
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [e2]},
        contexts=[p, r2])

    with self.subTest('Pipeline'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertLen(resolved, 2)
      self.assertEqual({a.id for a in resolved}, {e1.id, e2.id})

    with self.subTest('Pipeline + PipelineRun'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        context_queries {
          type {
            name: "pipeline_run"
          }
          name {
            field_value {
              string_value: "run-001"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertLen(resolved, 1)
      self.assertEqual(resolved[0].id, e1.id)

    with self.subTest('Pipeline + PipelineRun + Else'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        context_queries {
          type {
            name: "pipeline_run"
          }
          name {
            field_value {
              string_value: "run-001"
            }
          }
        }
        context_queries {
          type {
            name: "hahaha"
          }
          name {
            field_value {
              string_value: "i-am-a-troll"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertEmpty(resolved)

  def testResolveSingleChannel_OutputKey(self):
    p = self.put_context('pipeline', 'my-pipeline')
    e1 = self.put_artifact('Examples')
    e2 = self.put_artifact('Examples')
    self.put_execution(
        'CustomExampleGen',
        inputs={},
        outputs={'first': [e1], 'second': [e2]},
        contexts=[p])

    with self.subTest('Correct output_key'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
        output_key: "first"
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertLen(resolved, 1)
      self.assertEqual(resolved[0].id, e1.id)

    with self.subTest('Wrong output_key'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
        output_key: "third"
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertEmpty(resolved)

    with self.subTest('No output_key -> merged'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            name: "Examples"
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertEqual({a.id for a in resolved}, {e1.id, e2.id})

  def testResolveSingleChannel_BadArtifactQuery(self):
    p = self.put_context('pipeline', 'my-pipeline')
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [self.put_artifact('Examples')]},
        contexts=[p])

    with self.subTest('No type'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {}
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(
            self.mlmd_handle, ch)

    with self.subTest('No type.name'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            id: 123
          }
        }
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(
            self.mlmd_handle, ch)

    with self.subTest('Non-existential'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
          name {
            field_value {
              string_value: "my-pipeline"
            }
          }
        }
        artifact_query {
          type {
            name: "i dont exist"
          }
        }
      """)
      resolved = channel_resolver.resolve_single_channel(
          self.mlmd_handle, ch)
      self.assertEmpty(resolved)

  def testResolveSingleChannel_NoExecutions(self):
    self.put_context('pipeline', 'my-pipeline')
    ch = self.make_channel_spec("""
      context_queries {
        type {
          name: "pipeline"
        }
        name {
          field_value {
            string_value: "my-pipeline"
          }
        }
      }
      artifact_query {
        type {
          name: "Examples"
        }
      }
    """)
    resolved = channel_resolver.resolve_single_channel(
        self.mlmd_handle, ch)
    self.assertEmpty(resolved)

  def testResolveSingleChannel_NoArtifacts(self):
    p = self.put_context('pipeline', 'my-pipeline')
    self.put_execution('Dummy', inputs={}, outputs={}, contexts=[p])
    ch = self.make_channel_spec("""
      context_queries {
        type {
          name: "pipeline"
        }
        name {
          field_value {
            string_value: "my-pipeline"
          }
        }
      }
      artifact_query {
        type {
          name: "Examples"
        }
      }
    """)
    resolved = channel_resolver.resolve_single_channel(
        self.mlmd_handle, ch)
    self.assertEmpty(resolved)

  def testResolveUnionChannels_Deduplication(self):
    p = self.put_context('pipeline', 'my-pipeline')
    e1 = self.put_artifact('Examples')
    self.put_execution(
        'ExampleGen',
        inputs={},
        outputs={'examples': [e1]},
        contexts=[p])

    ch = self.make_channel_spec("""
      context_queries {
        type {
          name: "pipeline"
        }
        name {
          field_value {
            string_value: "my-pipeline"
          }
        }
      }
      artifact_query {
        type {
          name: "Examples"
        }
      }
      output_key: "examples"
    """)
    resolved = channel_resolver.resolve_union_channels(
        self.mlmd_handle, [ch, ch])
    self.assertLen(resolved, 1)
    self.assertEqual(resolved[0].id, e1.id)


if __name__ == '__main__':
  tf.test.main()
