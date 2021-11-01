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
"""Tests for tfx.dsl.components.common.importer."""

import tensorflow as tf
from tfx import types
from tfx.dsl.components.common import importer
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils

from ml_metadata.proto import metadata_store_pb2


class ImporterTest(tf.test.TestCase):

  def testImporterDefinitionWithSingleUri(self):
    impt = importer.Importer(
        source_uri='m/y/u/r/i',
        properties={
            'split_names': '["train", "eval"]',
        },
        custom_properties={
            'str_custom_property': 'abc',
            'int_custom_property': 123,
        },
        artifact_type=standard_artifacts.Examples).with_id('my_importer')
    self.assertDictEqual(
        impt.exec_properties, {
            importer.SOURCE_URI_KEY: 'm/y/u/r/i',
            importer.REIMPORT_OPTION_KEY: 0,
        })
    self.assertEmpty(impt.inputs)
    output_channel = impt.outputs[importer.IMPORT_RESULT_KEY]
    self.assertEqual(output_channel.type, standard_artifacts.Examples)
    # Tests properties in channel.
    self.assertEqual(output_channel.additional_properties, {
        'split_names': '["train", "eval"]',
    })
    self.assertEqual(output_channel.additional_custom_properties, {
        'str_custom_property': 'abc',
        'int_custom_property': 123,
    })
    # Tests properties in artifact.
    output_artifact = list(output_channel.get())[0]
    self.assertEqual(output_artifact.split_names, '["train", "eval"]')
    self.assertEqual(
        output_artifact.get_string_custom_property('str_custom_property'),
        'abc')
    self.assertEqual(
        output_artifact.get_int_custom_property('int_custom_property'), 123)

  def testImporterDumpsJsonRoundtrip(self):
    component_id = 'my_importer'
    source_uris = ['m/y/u/r/i']
    impt = importer.Importer(
        source_uri=source_uris,
        artifact_type=standard_artifacts.Examples).with_id(component_id)

    # The following line will raise an assertion if object not JSONable.
    json_text = json_utils.dumps(impt)

    actual_obj = json_utils.loads(json_text)
    self.assertEqual(actual_obj.id, component_id)
    self.assertEqual(actual_obj._source_uri, source_uris)


class ImporterDriverTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.connection_config = metadata_store_pb2.ConnectionConfig()
    self.connection_config.sqlite.SetInParent()
    self.properties = {
        'split_names': artifact_utils.encode_split_names(['train', 'eval'])
    }
    self.custom_properties = {
        'string_custom_property': 'abc',
        'int_custom_property': 123,
    }
    self.output_dict = {
        importer.IMPORT_RESULT_KEY:
            types.Channel(
                type=standard_artifacts.Examples,
                additional_properties=self.properties,
                additional_custom_properties=self.custom_properties)
    }
    self.source_uri = 'm/y/u/r/i'

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
      driver = importer.ImporterDriver(metadata_handler=m)
      execution_result = driver.pre_execution(
          component_info=self.component_info,
          pipeline_info=self.pipeline_info,
          driver_args=self.driver_args,
          input_dict={},
          output_dict=self.output_dict,
          exec_properties={
              importer.SOURCE_URI_KEY: self.source_uri,
              importer.REIMPORT_OPTION_KEY: int(reimport),
          })
      self.assertFalse(execution_result.use_cached_results)
      self.assertEmpty(execution_result.input_dict)
      self.assertEqual(
          1, len(execution_result.output_dict[importer.IMPORT_RESULT_KEY]))
      self.assertEqual(
          execution_result.output_dict[importer.IMPORT_RESULT_KEY][0].uri,
          self.source_uri)

      self.assertNotEmpty(self.output_dict[importer.IMPORT_RESULT_KEY].get())

      results = self.output_dict[importer.IMPORT_RESULT_KEY].get()
      self.assertEqual(1, len(results))
      result = results[0]
      self.assertEqual(result.uri, result.uri)
      for key, value in self.properties.items():
        self.assertEqual(value, getattr(result, key))
      for key, value in self.custom_properties.items():
        if isinstance(value, int):
          self.assertEqual(value, result.get_int_custom_property(key))
        elif isinstance(value, (str, bytes)):
          self.assertEqual(value, result.get_string_custom_property(key))
        else:
          raise ValueError('Invalid custom property value: %r.' % value)

  def testImportArtifact(self):
    self._callImporterDriver(reimport=True)

  def testReuseArtifact(self):
    self._callImporterDriver(reimport=False)


if __name__ == '__main__':
  tf.test.main()
