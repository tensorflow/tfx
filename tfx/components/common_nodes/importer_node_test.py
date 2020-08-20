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
"""Tests for tfx.components.common_nodes.importer_node."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.common_nodes import importer_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils


class ImporterNodeTest(tf.test.TestCase):

  def testImporterDefinitionWithSingleUri(self):
    impt = importer_node.ImporterNode(
        instance_name='my_importer',
        source_uri='m/y/u/r/i',
        properties={
            'split_names': '["train", "eval"]',
        },
        custom_properties={
            'str_custom_property': 'abc',
            'int_custom_property': 123,
        },
        artifact_type=standard_artifacts.Examples)
    self.assertDictEqual(
        impt.exec_properties, {
            importer_node.SOURCE_URI_KEY: 'm/y/u/r/i',
            importer_node.REIMPORT_OPTION_KEY: False,
            importer_node.PROPERTIES_KEY: {
                'split_names': '["train", "eval"]',
            },
            importer_node.CUSTOM_PROPERTIES_KEY: {
                'str_custom_property': 'abc',
                'int_custom_property': 123,
            },
        })
    self.assertEmpty(impt.inputs.get_all())
    self.assertEqual(impt.outputs[importer_node.IMPORT_RESULT_KEY].type,
                     standard_artifacts.Examples)

  def testImporterNodeDumpsJsonRoundtrip(self):
    instance_name = 'my_importer'
    source_uris = ['m/y/u/r/i']
    impt = importer_node.ImporterNode(
        instance_name=instance_name,
        source_uri=source_uris,
        artifact_type=standard_artifacts.Examples)

    # The following line will raise an assertion if object not JSONable.
    json_text = json_utils.dumps(impt)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(actual_obj._instance_name, instance_name)
    self.assertEqual(actual_obj._source_uri, source_uris)


class ImporterDriverTest(tf.test.TestCase):

  def setUp(self):
    super(ImporterDriverTest, self).setUp()
    self.connection_config = metadata_store_pb2.ConnectionConfig()
    self.connection_config.sqlite.SetInParent()
    self.output_dict = {
        importer_node.IMPORT_RESULT_KEY:
            types.Channel(type=standard_artifacts.Examples)
    }
    self.source_uri = 'm/y/u/r/i'
    self.properties = {
        'split_names': artifact_utils.encode_split_names(['train', 'eval'])
    }
    self.custom_properties = {
        'string_custom_property': 'abc',
        'int_custom_property': 123,
    }

    self.existing_artifacts = []
    existing_artifact = standard_artifacts.Examples()
    existing_artifact.uri = self.source_uri
    existing_artifact.split_names = self.properties['split_names']
    self.existing_artifacts.append(existing_artifact)

    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name='p_name', pipeline_root='p_root', run_id='run_id')
    self.component_info = data_types.ComponentInfo(
        component_type='c_type',
        component_id='c_id',
        pipeline_info=self.pipeline_info)
    self.driver_args = data_types.DriverArgs(enable_cache=True)

  def _callImporterDriver(self, reimport: bool):
    with metadata.Metadata(connection_config=self.connection_config) as m:
      m.publish_artifacts(self.existing_artifacts)
      driver = importer_node.ImporterDriver(metadata_handler=m)
      execution_result = driver.pre_execution(
          component_info=self.component_info,
          pipeline_info=self.pipeline_info,
          driver_args=self.driver_args,
          input_dict={},
          output_dict=self.output_dict,
          exec_properties={
              importer_node.SOURCE_URI_KEY: self.source_uri,
              importer_node.REIMPORT_OPTION_KEY: reimport,
              importer_node.PROPERTIES_KEY: self.properties,
              importer_node.CUSTOM_PROPERTIES_KEY: self.custom_properties,
          })
      self.assertFalse(execution_result.use_cached_results)
      self.assertEmpty(execution_result.input_dict)
      self.assertEqual(
          1, len(execution_result.output_dict[importer_node.IMPORT_RESULT_KEY]))
      self.assertEqual(
          execution_result.output_dict[importer_node.IMPORT_RESULT_KEY][0].uri,
          self.source_uri)

      self.assertNotEmpty(
          self.output_dict[importer_node.IMPORT_RESULT_KEY].get())

      results = self.output_dict[importer_node.IMPORT_RESULT_KEY].get()
      self.assertEqual(1, len(results))
      result = results[0]
      self.assertEqual(result.uri, result.uri)
      for key, value in self.properties.items():
        self.assertEqual(value, getattr(result, key))
      for key, value in self.custom_properties.items():
        if isinstance(value, int):
          self.assertEqual(value, result.get_int_custom_property(key))
        elif isinstance(value, (Text, bytes)):
          self.assertEqual(value, result.get_string_custom_property(key))
        else:
          raise ValueError('Invalid custom property value: %r.' % value)

  def testImportArtifact(self):
    self._callImporterDriver(reimport=True)

  def testReuseArtifact(self):
    self._callImporterDriver(reimport=False)


if __name__ == '__main__':
  tf.test.main()
