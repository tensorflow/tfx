# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

# Standard Imports

import tensorflow as tf
from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types.artifact import ArtifactState

from ml_metadata.proto import metadata_store_pb2


class MetadataTest(tf.test.TestCase):

  def setUp(self):
    super(MetadataTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._pipeline_info2 = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id2')
    self._pipeline_info3 = data_types.PipelineInfo(
        pipeline_name='my_pipeline2', pipeline_root='/tmp', run_id='my_run_id')
    self._pipeline_info4 = data_types.PipelineInfo(
        pipeline_name='my_pipeline2', pipeline_root='/tmp', run_id='my_run_id2')
    self._pipeline_info5 = data_types.PipelineInfo(
        pipeline_name='my_pipeline3', pipeline_root='/tmp', run_id='my_run_id3')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)
    self._component_info2 = data_types.ComponentInfo(
        component_type='a.b.d',
        component_id='my_component_2',
        pipeline_info=self._pipeline_info)
    self._component_info3 = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info3)
    self._component_info5 = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info5)

  def _check_artifact_state(self, metadata_handler: metadata.Metadata,
                            target: types.Artifact, state: Text):
    [artifact] = metadata_handler.store.get_artifacts_by_id([target.id])
    if 'state' in artifact.properties:
      current_artifact_state = artifact.properties['state'].string_value
    else:
      # This is for forward compatible for the artifact type cleanup.
      current_artifact_state = artifact.custom_properties['state'].string_value
    self.assertEqual(current_artifact_state, state)

  def _get_all_runs(self, metadata_handler: metadata.Metadata,
                    pipeline_name: Text):
    result = []
    for context in metadata_handler.store.get_contexts_by_type(
        metadata._CONTEXT_TYPE_PIPELINE_RUN):
      if context.properties['pipeline_name'].string_value == pipeline_name:
        result.append(context.properties['run_id'].string_value)
    return result

  def _get_execution_states(self, metadata_handler: metadata.Metadata,
                            pipeline_info: data_types.PipelineInfo):
    pipeline_run_context = metadata_handler.store.get_context_by_type_and_name(
        metadata._CONTEXT_TYPE_PIPELINE_RUN,
        pipeline_info.pipeline_run_context_name)
    result = {}
    if not pipeline_run_context:
      return result
    for execution in metadata_handler.store.get_executions_by_context(
        pipeline_run_context.id):
      result[execution.properties['component_id']
             .string_value] = execution.properties['state'].string_value
    return result

  def testArtifact(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self.assertListEqual([], m.store.get_artifacts())

      # Test publish artifact.
      artifact = standard_artifacts.Examples()
      artifact.uri = 'uri'
      artifact.split_names = artifact_utils.encode_split_names(
          ['train', 'eval'])
      m.publish_artifacts([artifact])
      [artifact] = m.store.get_artifacts()
      # Skip verifying time sensitive fields.
      artifact.ClearField('create_time_since_epoch')
      artifact.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """id: 1
        type_id: 1
        uri: "uri"
        properties {
          key: "split_names"
          value {
            string_value: "[\\"train\\", \\"eval\\"]"
          }
        }
        custom_properties {
          key: "state"
          value {
            string_value: "published"
          }
        }
        state: LIVE
        """, artifact)

      # Test get artifact.
      [artifact] = m.store.get_artifacts()
      self.assertListEqual([artifact], m.get_artifacts_by_uri('uri'))
      self.assertListEqual([artifact],
                           m.get_artifacts_by_type(
                               standard_artifacts.Examples.TYPE_NAME))

      # Test artifact state.
      self.assertEqual(artifact.state, metadata_store_pb2.Artifact.LIVE)
      self._check_artifact_state(m, artifact, ArtifactState.PUBLISHED)
      m.update_artifact_state(artifact, ArtifactState.DELETED)
      self._check_artifact_state(m, artifact, ArtifactState.DELETED)

  def testArtifactTypeRegistrationForwardCompatible(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self.assertListEqual([], m.store.get_artifacts())

      # Test publish some artifact, the Examples type is registered dynamically.
      artifact = standard_artifacts.Examples()
      m.publish_artifacts([artifact])
      artifact_type = m.store.get_artifact_type(type_name='Examples')
      self.assertProtoEquals(
          """id: 1
        name: "Examples"
        properties {
          key: "span"
          value: INT
        }
        properties {
          key: "version"
          value: INT
        }
        properties {
          key: "split_names"
          value: STRING
        }
        """, artifact_type)

      # Now mimic a future type updates registered by jobs of newer release
      artifact_type.properties['new_property'] = metadata_store_pb2.DOUBLE
      m.store.put_artifact_type(artifact_type, can_add_fields=True)

      # The artifact from the current artifacts can still be inserted.
      artifact2 = standard_artifacts.Examples()
      m.publish_artifacts([artifact2])
      stored_type = m.store.get_artifact_type(type_name='Examples')
      self.assertProtoEquals(
          """id: 1
        name: "Examples"
        properties {
          key: "span"
          value: INT
        }
        properties {
          key: "version"
          value: INT
        }
        properties {
          key: "split_names"
          value: STRING
        }
        properties {
          key: "new_property"
          value: DOUBLE
        }
        """, stored_type)
      self.assertEqual(2, len(m.store.get_artifacts()))

  def testExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      # Test prepare_execution.
      exec_properties = {'arg_one': 1}
      input_artifact = standard_artifacts.Examples()
      output_artifact = standard_artifacts.Examples()
      input_artifacts = {'input': [input_artifact]}
      output_artifacts = {'output': [output_artifact]}
      m.register_execution(
          input_artifacts=input_artifacts,
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      [execution] = m.store.get_executions_by_context(contexts[0].id)
      # Skip verifying time sensitive fields.
      execution.ClearField('create_time_since_epoch')
      execution.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 1
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }""", execution)

      # Test publish_execution.
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts=output_artifacts)
      # Make sure artifacts in output_dict are published.
      self.assertEqual(ArtifactState.PUBLISHED, output_artifact.state)
      # Make sure execution state are changed.
      [execution] = m.store.get_executions_by_id([execution.id])
      self.assertEqual(metadata.EXECUTION_STATE_COMPLETE,
                       execution.properties['state'].string_value)
      # Make sure events are published.
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertEqual(2, len(events))
      self.assertEqual(input_artifact.id, events[0].artifact_id)
      self.assertEqual(metadata_store_pb2.Event.INPUT, events[0].type)
      self.assertProtoEquals(
          """
          steps {
            key: "input"
          }
          steps {
            index: 0
          }""", events[0].path)
      self.assertEqual(output_artifact.id, events[1].artifact_id)
      self.assertEqual(metadata_store_pb2.Event.OUTPUT, events[1].type)
      self.assertProtoEquals(
          """
          steps {
            key: "output"
          }
          steps {
            index: 0
          }""", events[1].path)

  def testRegisterExecutionUpdatedExecutionType(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts_one = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info)
      contexts_two = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info3)

      # Puts in execution with less columns needed in MLMD schema first and
      # puts in execution with more columns needed next. Verifies the schema
      # update will not be breaking change.
      exec_properties_one = {'arg_one': 1}
      exec_properties_two = {'arg_one': 1, 'arg_two': 2}
      execution_one = m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties_one,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts_one)
      execution_two = m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties_two,
          pipeline_info=self._pipeline_info3,
          component_info=self._component_info3,
          contexts=contexts_two)
      [execution_one, execution_two
      ] = m.store.get_executions_by_id([execution_one.id, execution_two.id])
      # Skip verifying time sensitive fields.
      execution_one.ClearField('create_time_since_epoch')
      execution_one.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 1
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }""", execution_one)
      # Skip verifying time sensitive fields.
      execution_two.ClearField('create_time_since_epoch')
      execution_two.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 2
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline2"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }
        properties {
          key: "arg_two"
          value {
            string_value: "2"
          }
        }""", execution_two)

  def testRegisterExecutionIdempotency(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      m.register_execution(
          exec_properties={'a': 1},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      execution = m.register_execution(
          exec_properties={'a': 1},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      self.assertEqual(execution.id, 1)
      self.assertEqual(len(m.store.get_executions()), 1)

  def testRegisterExecutionBackwardCompatibility(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)

      # Puts in execution with more columns needed in MLMD schema first and
      # puts in execution with less columns needed next. Verifies the schema
      # update will not affect backward compatibility.
      exec_properties_one = {'arg_one': 1, 'arg_two': 2}
      exec_properties_two = {'arg_one': 1}
      exec_properties_three = {'arg_one': 1, 'arg_three': 3}
      execution_one = m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties_one,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      execution_two = m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties_two,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info3,
          contexts=contexts)
      execution_three = m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties_three,
          pipeline_info=self._pipeline_info5,
          component_info=self._component_info5,
          contexts=contexts)
      [execution_one, execution_two,
       execution_three] = m.store.get_executions_by_id(
           [execution_one.id, execution_two.id, execution_three.id])
      # Skip verifying time sensitive fields.
      execution_one.ClearField('create_time_since_epoch')
      execution_one.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 1
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }
        properties {
          key: "arg_two"
          value {
            string_value: "2"
          }
        }""", execution_one)
      # Skip verifying time sensitive fields.
      execution_two.ClearField('create_time_since_epoch')
      execution_two.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 2
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }""", execution_two)
      # Skip verifying time sensitive fields.
      execution_three.ClearField('create_time_since_epoch')
      execution_three.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(
          """
        id: 3
        type_id: 3
        last_known_state: RUNNING
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }
        properties {
          key: "pipeline_name"
          value {
            string_value: "my_pipeline3"
          }
        }
        properties {
          key: "pipeline_root"
          value {
            string_value: "/tmp"
          }
        }
        properties {
          key: "run_id"
          value {
            string_value: "my_run_id3"
          }
        }
        properties {
          key: "component_id"
          value {
            string_value: "my_component"
          }
        }
        properties {
          key: "arg_one"
          value {
            string_value: "1"
          }
        }
        properties {
          key: "arg_three"
          value {
            string_value: "3"
          }
        }""", execution_three)

  def testFetchPreviousResult(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:

      # Create an 'previous' execution.
      exec_properties = {'log_root': 'path'}
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      input_artifacts = {'input': [standard_artifacts.Examples()]}
      output_artifact = standard_artifacts.Examples()
      output_artifact.uri = 'my_uri'
      output_artifacts = {'output': [output_artifact]}
      m.register_execution(
          input_artifacts=input_artifacts,
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts=output_artifacts)

      # Test previous_run.
      self.assertIsNone(
          m.get_cached_outputs(
              input_artifacts={},
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=self._component_info))
      self.assertIsNone(
          m.get_cached_outputs(
              input_artifacts=input_artifacts,
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=data_types.ComponentInfo(
                  component_id='unique',
                  component_type='a.b.c',
                  pipeline_info=self._pipeline_info)))
      # Having the same set of input artifact ids, but duplicated.
      self.assertIsNone(
          m.get_cached_outputs(
              input_artifacts={
                  'input': input_artifacts['input'],
                  'another_input': input_artifacts['input']
              },
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=self._component_info))
      cached_output_artifacts = m.get_cached_outputs(
          input_artifacts=input_artifacts,
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      self.assertEqual(len(cached_output_artifacts), 1)
      self.assertEqual(len(cached_output_artifacts['output']), 1)
      cached_output_artifact = cached_output_artifacts['output'][
          0].mlmd_artifact
      # Skip verifying time sensitive fields.
      cached_output_artifact.ClearField('create_time_since_epoch')
      cached_output_artifact.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(cached_output_artifact,
                             output_artifact.mlmd_artifact)

  def testGetCachedOutputNoInput(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:

      # Create an 'previous' execution.
      exec_properties = {'log_root': 'path'}
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      output_artifact = standard_artifacts.Examples()
      output_artifact.uri = 'my_uri'
      output_artifacts = {'output': [output_artifact]}
      m.register_execution(
          input_artifacts={},
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts=output_artifacts)

      cached_output_artifacts = m.get_cached_outputs(
          input_artifacts={},
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      self.assertEqual(len(cached_output_artifacts), 1)
      self.assertEqual(len(cached_output_artifacts['output']), 1)
      cached_output_artifact = cached_output_artifacts['output'][
          0].mlmd_artifact
      # Skip verifying time sensitive fields.
      cached_output_artifact.ClearField('create_time_since_epoch')
      cached_output_artifact.ClearField('last_update_time_since_epoch')
      self.assertProtoEquals(cached_output_artifact,
                             output_artifact.mlmd_artifact)

  def testSearchArtifacts(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      exec_properties = {'log_root': 'path'}
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      m.register_execution(
          input_artifacts={'input': [standard_artifacts.Examples()]},
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      output_artifact = standard_artifacts.Examples()
      output_artifact.uri = 'my/uri'
      output_dict = {'output': [output_artifact]}
      m.publish_execution(
          component_info=self._component_info, output_artifacts=output_dict)
      [artifact] = m.search_artifacts(
          artifact_name='output',
          pipeline_info=self._pipeline_info,
          producer_component_id=self._component_info.component_id)
      self.assertEqual(artifact.uri, output_artifact.uri)

  def testPublishSkippedExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      exec_properties = {'log_root': 'path'}
      output_artifact = standard_artifacts.Examples()
      output_artifact.uri = 'my/uri'
      output_artifacts = {'output': [output_artifact]}
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      execution = m.register_execution(
          input_artifacts={'input': [standard_artifacts.Examples()]},
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      m.update_execution(
          execution=execution,
          component_info=self._component_info,
          output_artifacts=output_artifacts,
          execution_state=metadata.EXECUTION_STATE_CACHED,
          contexts=contexts)
      m.publish_execution(component_info=self._component_info)

  def testGetExecutionStates(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts_one = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info)
      contexts_two = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info)
      contexts_three = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info2)

      self.assertListEqual(
          [self._pipeline_info.run_id, self._pipeline_info2.run_id],
          self._get_all_runs(m, 'my_pipeline'))

      m.register_execution(
          input_artifacts={},
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts_one)
      m.publish_execution(component_info=self._component_info)
      m.register_execution(
          input_artifacts={},
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info2,
          contexts=contexts_two)
      m.register_execution(
          input_artifacts={},
          exec_properties={},
          pipeline_info=self._pipeline_info2,
          component_info=self._component_info3,
          contexts=contexts_three)
      states = self._get_execution_states(m, self._pipeline_info)
      self.assertDictEqual(
          {
              self._component_info.component_id:
                  metadata.EXECUTION_STATE_COMPLETE,
              self._component_info2.component_id:
                  metadata.EXECUTION_STATE_NEW,
          }, states)

  def testUpdateExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      m.register_execution(
          input_artifacts={},
          exec_properties={'k': 'v1'},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=contexts)
      [execution] = m.store.get_executions_by_context(
          m.get_component_run_context(self._component_info).id)
      self.assertEqual(execution.properties['k'].string_value, 'v1')
      self.assertEqual(execution.properties['state'].string_value,
                       metadata.EXECUTION_STATE_NEW)
      self.assertEqual(execution.last_known_state,
                       metadata_store_pb2.Execution.RUNNING)

      m.update_execution(
          execution,
          self._component_info,
          input_artifacts={'input_a': [standard_artifacts.Examples()]},
          exec_properties={'k': 'v2'},
          contexts=contexts)

      [execution] = m.store.get_executions_by_context(
          m.get_component_run_context(self._component_info).id)
      self.assertEqual(execution.properties['k'].string_value, 'v2')
      self.assertEqual(execution.properties['state'].string_value,
                       metadata.EXECUTION_STATE_NEW)
      self.assertEqual(execution.last_known_state,
                       metadata_store_pb2.Execution.RUNNING)
      [event] = m.store.get_events_by_execution_ids([execution.id])
      self.assertEqual(event.artifact_id, 1)
      [artifact] = m.store.get_artifacts_by_context(
          m.get_component_run_context(self._component_info).id)
      self.assertEqual(artifact.id, 1)

      aa = standard_artifacts.Examples()
      aa.set_mlmd_artifact(artifact)
      m.update_execution(
          execution, self._component_info, input_artifacts={'input_a': [aa]})
      [event] = m.store.get_events_by_execution_ids([execution.id])
      self.assertEqual(event.type, metadata_store_pb2.Event.INPUT)

      m.publish_execution(
          self._component_info,
          output_artifacts={'output': [standard_artifacts.Model()]},
          exec_properties={'k': 'v3'})

      [execution] = m.store.get_executions_by_context(
          m.get_component_run_context(self._component_info).id)
      self.assertEqual(execution.properties['k'].string_value, 'v3')
      self.assertEqual(execution.properties['state'].string_value,
                       metadata.EXECUTION_STATE_COMPLETE)
      self.assertEqual(execution.last_known_state,
                       metadata_store_pb2.Execution.COMPLETE)
      events = m.store.get_events_by_execution_ids([execution.id])
      self.assertLen(events, 2)
      [event_b] = (
          e for e in events if e.type == metadata_store_pb2.Event.OUTPUT)
      self.assertEqual(event_b.artifact_id, 2)
      artifacts = m.store.get_artifacts_by_context(
          m.get_component_run_context(self._component_info).id)
      self.assertLen(artifacts, 2)
      [artifact_b] = (a for a in artifacts if a.id == 2)
      self._check_artifact_state(m, artifact_b, ArtifactState.PUBLISHED)

  def testGetQualifiedArtifacts(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts_one = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info)
      contexts_two = m.register_pipeline_contexts_if_not_exists(
          self._pipeline_info3)
      # The first execution, with matched:
      #   - pipeline context
      #   - producer component id
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          contexts=list(contexts_one))
      # artifact_one will be output with matched artifact type and output key
      artifact_one = standard_artifacts.Model()
      # artifact_one will be output with matched artifact type only
      artifact_two = standard_artifacts.Model()
      m.publish_execution(
          component_info=self._component_info,
          output_artifacts={
              'k1': [artifact_one],
              'k2': [artifact_two]
          })
      # The second execution, with matched pipeline context only
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info2,
          contexts=list(contexts_one))
      # artifact_three will be output with matched artifact type and output key
      artifact_three = standard_artifacts.Model()
      m.publish_execution(
          component_info=self._component_info2,
          output_artifacts={'k1': [artifact_three]})
      # The third execution, with matched producer component id only
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info3,
          component_info=self._component_info3,
          contexts=list(contexts_two))
      # artifact_three will be output with matched artifact type and output key
      artifact_four = standard_artifacts.Model()
      m.publish_execution(
          component_info=self._component_info3,
          output_artifacts={'k1': [artifact_four]})

      result = m.get_qualified_artifacts(
          contexts=contexts_one,
          type_name=standard_artifacts.Model().type_name,
          producer_component_id=self._component_info.component_id,
          output_key='k1')
      self.assertEqual(len(result), 1)
      self.assertEqual(result[0].artifact.id, artifact_one.id)

  def testContext(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)
      # Duplicated call should succeed.
      contexts = m.register_pipeline_contexts_if_not_exists(self._pipeline_info)

      self.assertProtoEquals(
          """
          id: 1
          name: 'pipeline'
          properties {
            key: "pipeline_name"
            value: STRING
          }
          """, m.store.get_context_type(metadata._CONTEXT_TYPE_PIPELINE))
      self.assertProtoEquals(
          """
          id: 2
          name: 'run'
          properties {
            key: "pipeline_name"
            value: STRING
          }
          properties {
            key: "run_id"
            value: STRING
          }
          """, m.store.get_context_type(metadata._CONTEXT_TYPE_PIPELINE_RUN))
      self.assertEqual(len(contexts), 2)
      self.assertEqual(
          contexts[0],
          m.store.get_context_by_type_and_name(
              metadata._CONTEXT_TYPE_PIPELINE,
              self._pipeline_info.pipeline_context_name))
      self.assertEqual(
          contexts[1],
          m.store.get_context_by_type_and_name(
              metadata._CONTEXT_TYPE_PIPELINE_RUN,
              self._pipeline_info.pipeline_run_context_name))

  def testInvalidConnection(self):
    # read only connection to a unknown file
    invalid_config = metadata_store_pb2.ConnectionConfig()
    invalid_config.sqlite.filename_uri = 'unknown_file'
    invalid_config.sqlite.connection_mode = 1
    # test the runtime error contains detailed information
    with self.assertRaisesRegex(RuntimeError, 'unable to open database file'):
      with metadata.Metadata(connection_config=invalid_config) as m:
        m.store()


if __name__ == '__main__':
  tf.test.main()
