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

# Standard Imports
import mock
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types.artifact import ArtifactState


class MetadataTest(tf.test.TestCase):

  def setUp(self):
    super(MetadataTest, self).setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c', component_id='my_component')
    self._component_info2 = data_types.ComponentInfo(
        component_type='a.b.d', component_id='my_component_2')
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._pipeline_info2 = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id2')
    self._pipeline_info3 = data_types.PipelineInfo(
        pipeline_name='my_pipeline2', pipeline_root='/tmp', run_id='my_run_id')
    self._pipeline_info4 = data_types.PipelineInfo(
        pipeline_name='my_pipeline2', pipeline_root='/tmp', run_id='my_run_id2')

  def testEmptyArtifact(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      m.publish_artifacts([])
      eid = m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      m.publish_execution(eid, {}, {})
      [execution] = m.store.get_executions_by_id([eid])
      self.assertProtoEquals(
          """
        id: 1
        type_id: 1
        properties {
          key: "state"
          value {
            string_value: "complete"
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
        }""", execution)

  def testArtifact(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      self.assertListEqual([], m.get_all_artifacts())

      # Test publish artifact.
      artifact = standard_artifacts.Examples()
      artifact.uri = 'uri'
      artifact.split_names = artifact_utils.encode_split_names(
          ['train', 'eval'])
      m.publish_artifacts([artifact])
      [artifact] = m.store.get_artifacts()
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
        """, artifact)

      # Test get artifact.
      self.assertListEqual([artifact], m.get_all_artifacts())
      self.assertListEqual([artifact], m.get_artifacts_by_uri('uri'))
      self.assertListEqual([artifact], m.get_artifacts_by_type(
          standard_artifacts.Examples.TYPE_NAME))

      # Test artifact state.
      m.check_artifact_state(artifact, ArtifactState.PUBLISHED)
      m.update_artifact_state(artifact, ArtifactState.DELETED)
      m.check_artifact_state(artifact, ArtifactState.DELETED)
      self.assertRaises(RuntimeError, m.check_artifact_state, artifact,
                        ArtifactState.PUBLISHED)

  def testExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      context_id = m.register_run_context_if_not_exists(self._pipeline_info)

      # Test prepare_execution.
      exec_properties = {'arg_one': 1}
      eid = m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      [execution] = m.store.get_executions_by_context(context_id)
      self.assertProtoEquals(
          """
        id: 1
        type_id: 2
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
      input_artifact = standard_artifacts.Examples()
      m.publish_artifacts([input_artifact])
      output_artifact = standard_artifacts.Examples()
      input_dict = {'input': [input_artifact]}
      output_dict = {'output': [output_artifact]}
      m.publish_execution(eid, input_dict, output_dict)
      # Make sure artifacts in output_dict are published.
      self.assertEqual(ArtifactState.PUBLISHED, output_artifact.state)
      # Make sure execution state are changed.
      [execution] = m.store.get_executions_by_id([eid])
      self.assertEqual(metadata.EXECUTION_STATE_COMPLETE,
                       execution.properties['state'].string_value)
      # Make sure events are published.
      events = m.store.get_events_by_execution_ids([eid])
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
      context_id = m.register_run_context_if_not_exists(self._pipeline_info)

      # Puts in execution with less columns needed in MLMD schema first and
      # puts in execution with more columns needed next. Verifies the schema
      # update will not be breaking change.
      exec_properties_one = {'arg_one': 1}
      exec_properties_two = {'arg_one': 1, 'arg_two': 2}
      eid_one = m.register_execution(
          exec_properties=exec_properties_one,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      eid_two = m.register_execution(
          exec_properties=exec_properties_two,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      [execution_one,
       execution_two] = m.store.get_executions_by_id([eid_one, eid_two])
      self.assertProtoEquals(
          """
        id: 1
        type_id: 2
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
      self.assertProtoEquals(
          """
        id: 2
        type_id: 2
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
        }""", execution_two)

  def testRegisterExecutionBackwardCompatibility(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      context_id = m.register_run_context_if_not_exists(self._pipeline_info)

      # Puts in execution with more columns needed in MLMD schema first and
      # puts in execution with less columns needed next. Verifies the schema
      # update will not affect backward compatibility.
      exec_properties_one = {'arg_one': 1}
      exec_properties_two = {'arg_one': 1, 'arg_two': 2}
      eid_two = m.register_execution(
          exec_properties=exec_properties_two,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      eid_one = m.register_execution(
          exec_properties=exec_properties_one,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      [execution_one,
       execution_two] = m.store.get_executions_by_id([eid_one, eid_two])
      self.assertProtoEquals(
          """
        id: 2
        type_id: 2
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
      self.assertProtoEquals(
          """
        id: 1
        type_id: 2
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
        }""", execution_two)

  def testFetchPreviousResult(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:

      # Create an 'previous' execution.
      exec_properties = {'log_root': 'path'}
      eid = m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      input_artifact = standard_artifacts.Examples()
      m.publish_artifacts([input_artifact])
      output_artifact = standard_artifacts.Examples()
      input_artifacts = {'input': [input_artifact]}
      output_artifacts = {'output': [output_artifact]}
      m.publish_execution(eid, input_artifacts, output_artifacts)

      # Test previous_run.
      self.assertEqual(
          None,
          m.previous_execution(
              input_artifacts={},
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=self._component_info))
      self.assertEqual(
          None,
          m.previous_execution(
              input_artifacts=input_artifacts,
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=data_types.ComponentInfo(
                  component_id='unique', component_type='a.b.c')))
      self.assertEqual(
          eid,
          m.previous_execution(
              input_artifacts=input_artifacts,
              exec_properties=exec_properties,
              pipeline_info=self._pipeline_info,
              component_info=self._component_info))

      # Test fetch_previous_result_artifacts.
      new_output_artifact = standard_artifacts.Examples()
      self.assertNotEqual(ArtifactState.PUBLISHED,
                          new_output_artifact.state)
      new_output_dict = {'output': [new_output_artifact]}
      updated_output_dict = m.fetch_previous_result_artifacts(
          new_output_dict, eid)
      previous_artifact = output_artifacts['output'][-1].mlmd_artifact
      current_artifact = updated_output_dict['output'][-1].mlmd_artifact
      self.assertEqual(ArtifactState.PUBLISHED,
                       current_artifact.custom_properties['state'].string_value)
      self.assertEqual(previous_artifact.id, current_artifact.id)
      self.assertEqual(previous_artifact.type_id, current_artifact.type_id)

  def testGetCachedExecutionIds(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      mock_store = mock.Mock()
      mock_store.get_events_by_execution_ids.side_effect = [
          [
              metadata_store_pb2.Event(
                  artifact_id=1, type=metadata_store_pb2.Event.INPUT)
          ],
          [
              metadata_store_pb2.Event(
                  artifact_id=1, type=metadata_store_pb2.Event.INPUT),
              metadata_store_pb2.Event(
                  artifact_id=2, type=metadata_store_pb2.Event.INPUT),
              metadata_store_pb2.Event(
                  artifact_id=3, type=metadata_store_pb2.Event.INPUT)
          ],
          [
              metadata_store_pb2.Event(
                  artifact_id=1, type=metadata_store_pb2.Event.INPUT),
              metadata_store_pb2.Event(
                  artifact_id=2, type=metadata_store_pb2.Event.INPUT),
          ],
      ]
      m._store = mock_store

      input_one = standard_artifacts.Examples()
      input_one.id = 1
      input_two = standard_artifacts.Examples()
      input_two.id = 2

      input_dict = {
          'input_one': [input_one],
          'input_two': [input_two],
      }

      self.assertEqual(1, m._get_cached_execution_id(input_dict, [3, 2, 1]))

  def testSearchArtifacts(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      exec_properties = {'log_root': 'path'}
      eid = m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      input_artifact = standard_artifacts.Examples()
      m.publish_artifacts([input_artifact])
      output_artifact = types.Artifact(type_name='MyOutputArtifact')
      output_artifact.uri = 'my/uri'
      input_dict = {'input': [input_artifact]}
      output_dict = {'output': [output_artifact]}
      m.publish_execution(eid, input_dict, output_dict)
      [artifact] = m.search_artifacts(
          artifact_name='output',
          pipeline_name=self._pipeline_info.pipeline_name,
          run_id=self._pipeline_info.run_id,
          producer_component_id=self._component_info.component_id)
      self.assertEqual(artifact.uri, output_artifact.uri)

  def testPublishSkippedExecution(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      exec_properties = {'log_root': 'path'}
      eid = m.register_execution(
          exec_properties=exec_properties,
          pipeline_info=self._pipeline_info,
          component_info=self._component_info)
      input_artifact = standard_artifacts.Examples()
      m.publish_artifacts([input_artifact])
      output_artifact = types.Artifact(type_name='MyOutputArtifact')
      output_artifact.uri = 'my/uri'
      [published_artifact] = m.publish_artifacts([output_artifact])
      output_artifact.set_mlmd_artifact(published_artifact)
      input_dict = {'input': [input_artifact]}
      output_dict = {'output': [output_artifact]}
      m.publish_execution(
          eid, input_dict, output_dict, state=metadata.EXECUTION_STATE_CACHED)

  def testGetExecutionStates(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      context_id = m.register_run_context_if_not_exists(self._pipeline_info)
      context_id2 = m.register_run_context_if_not_exists(self._pipeline_info2)

      self.assertListEqual(
          [self._pipeline_info.run_id, self._pipeline_info2.run_id],
          m.get_all_runs('my_pipeline'))

      eid = m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info,
          run_context_id=context_id)
      m.publish_execution(eid, {}, {})
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info,
          component_info=self._component_info2,
          run_context_id=context_id)
      m.register_execution(
          exec_properties={},
          pipeline_info=self._pipeline_info2,
          component_info=self._component_info,
          run_context_id=context_id2)
      states = m.get_execution_states(self._pipeline_info)
      self.assertDictEqual(
          {
              self._component_info.component_id:
                  metadata.EXECUTION_STATE_COMPLETE,
              self._component_info2.component_id:
                  metadata.EXECUTION_STATE_NEW,
          }, states)

  def testContext(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:

      cid1 = m.register_run_context_if_not_exists(self._pipeline_info)
      cid2 = m.register_run_context_if_not_exists(self._pipeline_info2)
      cid3 = m.register_run_context_if_not_exists(self._pipeline_info3)

      context_type = m.store.get_context_type('run')
      self.assertProtoEquals(
          """
          id: 1
          name: 'run'
          properties {
            key: "pipeline_name"
            value: STRING
          }
          properties {
            key: "run_id"
            value: STRING
          }
          """, context_type)
      [context] = m.store.get_contexts_by_id([cid1])
      self.assertProtoEquals(
          """
          id: 1
          type_id: 1
          name: 'my_pipeline.my_run_id'
          properties {
            key: "pipeline_name"
            value {
              string_value: "my_pipeline"
            }
          }
          properties {
            key: "run_id"
            value {
              string_value: "my_run_id"
            }
          }
          """, context)

      self.assertEqual(
          cid1, m.register_run_context_if_not_exists(self._pipeline_info))
      self.assertEqual(cid1, m._get_run_context_id(self._pipeline_info))
      self.assertEqual(cid2, m._get_run_context_id(self._pipeline_info2))
      self.assertEqual(cid3, m._get_run_context_id(self._pipeline_info3))
      self.assertEqual(None, m._get_run_context_id(self._pipeline_info4))


if __name__ == '__main__':
  tf.test.main()
