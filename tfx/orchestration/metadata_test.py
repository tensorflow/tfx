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

import os
# Standard Imports
import mock
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.metadata import Metadata
from tfx.utils import logging_utils
from tfx.utils import types


class MetadataTest(tf.test.TestCase):

  def setUp(self):
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    log_root = os.path.join(self.get_temp_dir(), 'log_dir')
    logger_config = logging_utils.LoggerConfig(log_root=log_root)
    self._logger = logging_utils.get_logger(logger_config)

  def test_empty_artifact(self):
    with Metadata(
        connection_config=self._connection_config, logger=self._logger) as m:
      m.publish_artifacts([])
      eid = m.prepare_execution('Test', {})
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
        }""", execution)

  def test_artifact(self):
    with Metadata(
        connection_config=self._connection_config, logger=self._logger) as m:
      self.assertListEqual([], m.get_all_artifacts())

      # Test publish artifact.
      artifact = types.TfxType(type_name='ExamplesPath')
      m.publish_artifacts([artifact])
      [artifact] = m.store.get_artifacts()
      self.assertProtoEquals(
          """id: 1
        type_id: 1
        uri: ""
        properties {
          key: "split"
          value {
            string_value: ""
          }
        }
        properties {
          key: "state"
          value {
            string_value: "published"
          }
        }
        properties {
          key: "type_name"
          value {
            string_value: "ExamplesPath"
          }
        }""", artifact)

      # Test get artifact.
      self.assertListEqual([artifact], m.get_all_artifacts())

      # Test artifact state.
      m.check_artifact_state(artifact, types.ARTIFACT_STATE_PUBLISHED)
      m.update_artifact_state(artifact, types.ARTIFACT_STATE_DELETED)
      m.check_artifact_state(artifact, types.ARTIFACT_STATE_DELETED)
      self.assertRaises(RuntimeError, m.check_artifact_state, artifact,
                        types.ARTIFACT_STATE_PUBLISHED)

  def test_execution(self):
    with Metadata(
        connection_config=self._connection_config, logger=self._logger) as m:

      # Test prepare_execution.
      exec_properties = {}
      eid = m.prepare_execution('Test', exec_properties)
      [execution] = m.store.get_executions()
      self.assertProtoEquals(
          """
        id: 1
        type_id: 1
        properties {
          key: "state"
          value {
            string_value: "new"
          }
        }""", execution)

      # Test publish_execution.
      input_artifact = types.TfxType(type_name='ExamplesPath')
      m.publish_artifacts([input_artifact])
      output_artifact = types.TfxType(type_name='ExamplesPath')
      input_dict = {'input': [input_artifact]}
      output_dict = {'output': [output_artifact]}
      m.publish_execution(eid, input_dict, output_dict)
      # Make sure artifacts in output_dict are published.
      self.assertEqual(types.ARTIFACT_STATE_PUBLISHED, output_artifact.state)
      # Make sure execution state are changed.
      [execution] = m.store.get_executions_by_id([eid])
      self.assertEqual('complete', execution.properties['state'].string_value)
      # Make sure events are published.
      events = m.store.get_events_by_execution_ids([eid])
      self.assertEqual(2, len(events))
      self.assertEqual(input_artifact.id, events[0].artifact_id)
      self.assertEqual(metadata_store_pb2.Event.DECLARED_INPUT, events[0].type)
      self.assertProtoEquals(
          """
          steps {
            key: "input"
          }
          steps {
            index: 0
          }""", events[0].path)
      self.assertEqual(output_artifact.id, events[1].artifact_id)
      self.assertEqual(metadata_store_pb2.Event.DECLARED_OUTPUT, events[1].type)
      self.assertProtoEquals(
          """
          steps {
            key: "output"
          }
          steps {
            index: 0
          }""", events[1].path)

  def test_fetch_previous_result(self):
    with Metadata(
        connection_config=self._connection_config, logger=self._logger) as m:

      # Create an 'previous' execution.
      exec_properties = {'log_root': 'path'}
      eid = m.prepare_execution('Test', exec_properties)
      input_artifact = types.TfxType(type_name='ExamplesPath')
      m.publish_artifacts([input_artifact])
      output_artifact = types.TfxType(type_name='ExamplesPath')
      input_dict = {'input': [input_artifact]}
      output_dict = {'output': [output_artifact]}
      m.publish_execution(eid, input_dict, output_dict)

      # Test previous_run.
      self.assertEqual(None, m.previous_run('Test', input_dict, {}))
      self.assertEqual(None, m.previous_run('Test', {}, exec_properties))
      self.assertEqual(None, m.previous_run('Test2', input_dict,
                                            exec_properties))
      self.assertEqual(eid, m.previous_run('Test', input_dict, exec_properties))

      # Test fetch_previous_result_artifacts.
      new_output_artifact = types.TfxType(type_name='ExamplesPath')
      self.assertNotEqual(types.ARTIFACT_STATE_PUBLISHED,
                          new_output_artifact.state)
      new_output_dict = {'output': [new_output_artifact]}
      updated_output_dict = m.fetch_previous_result_artifacts(
          new_output_dict, eid)
      previous_artifact = output_dict['output'][-1].artifact
      current_artifact = updated_output_dict['output'][-1].artifact
      self.assertEqual(types.ARTIFACT_STATE_PUBLISHED,
                       current_artifact.properties['state'].string_value)
      self.assertEqual(previous_artifact.id, current_artifact.id)
      self.assertEqual(previous_artifact.type_id, current_artifact.type_id)

  def test_get_cached_execution_ids(self):
    with Metadata(
        connection_config=self._connection_config, logger=self._logger) as m:
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

      input_one = types.TfxType(type_name='ExamplesPath')
      input_one.id = 1
      input_two = types.TfxType(type_name='ExamplesPath')
      input_two.id = 2

      input_dict = {
          'input_one': [input_one],
          'input_two': [input_two],
      }

      self.assertEqual(1, m._get_cached_execution_id(input_dict, [3, 2, 1]))


if __name__ == '__main__':
  tf.test.main()
