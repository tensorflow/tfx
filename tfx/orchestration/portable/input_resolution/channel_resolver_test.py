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
from tfx.orchestration.portable.mlmd import event_lib
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


class MlmdMixins:

  def init_mlmd(self):
    config = metadata_store_pb2.ConnectionConfig()
    config.fake_database.SetInParent()
    self.store = mlmd.MetadataStore(config)
    self.__context_type_ids = {}
    self.__artifact_type_ids = {}
    self.__execution_type_ids = {}

  def _get_context_type_id(self, type_name: str):
    if type_name not in self.__context_type_ids:
      result = self.store.put_context_type(
          metadata_store_pb2.ContextType(name=type_name))
      self.__context_type_ids[type_name] = result
    return self.__context_type_ids[type_name]

  def put_context(self, context_type: str, context_name: str):
    result = metadata_store_pb2.Context(
        type_id=self._get_context_type_id(context_type),
        name=context_name)
    result.id = self.store.put_contexts([result])[0]
    return result

  def _get_artifact_type_id(self, type_name: str):
    if type_name not in self.__artifact_type_ids:
      result = self.store.put_artifact_type(
          metadata_store_pb2.ArtifactType(name=type_name))
      self.__artifact_type_ids[type_name] = result
    return self.__artifact_type_ids[type_name]

  def put_artifact(self, artifact_type: str, uri: str = '/fake'):
    result = metadata_store_pb2.Artifact(
        type_id=self._get_artifact_type_id(artifact_type),
        uri=uri, state=metadata_store_pb2.Artifact.LIVE)
    result.id = self.store.put_artifacts([result])[0]
    return result

  def _get_execution_type_id(self, type_name: str):
    if type_name not in self.__execution_type_ids:
      result = self.store.put_execution_type(
          metadata_store_pb2.ExecutionType(name=type_name))
      self.__execution_type_ids[type_name] = result
    return self.__execution_type_ids[type_name]

  def put_execution(self, execution_type: str, inputs, outputs, contexts):
    result = metadata_store_pb2.Execution(
        type_id=self._get_execution_type_id(execution_type),
        last_known_state=metadata_store_pb2.Execution.COMPLETE)
    artifact_and_events = []
    for input_key, artifacts in inputs.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(
            metadata_store_pb2.Event.INPUT, input_key, i)
        artifact_and_events.append((artifact, event))
    for output_key, artifacts in outputs.items():
      for i, artifact in enumerate(artifacts):
        event = event_lib.generate_event(
            metadata_store_pb2.Event.OUTPUT, output_key, i)
        artifact_and_events.append((artifact, event))
    result.id = self.store.put_execution(
        result, artifact_and_events, contexts)[0]
    return result


class ChannelResolverTest(tf.test.TestCase, MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()

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
    resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
        channel_resolver.resolve_single_channel(self.store, ch)

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
        channel_resolver.resolve_single_channel(self.store, ch)

    with self.subTest('No name'):
      ch = self.make_channel_spec("""
        context_queries {
          type {
            name: "pipeline"
          }
        }
      """)
      with self.assertRaises(ValueError):
        channel_resolver.resolve_single_channel(self.store, ch)

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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
        channel_resolver.resolve_single_channel(self.store, ch)

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
        channel_resolver.resolve_single_channel(self.store, ch)

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
      resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
    resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
    resolved = channel_resolver.resolve_single_channel(self.store, ch)
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
    resolved = channel_resolver.resolve_union_channels(self.store, [ch, ch])
    self.assertLen(resolved, 1)
    self.assertEqual(resolved[0].id, e1.id)


if __name__ == '__main__':
  tf.test.main()
