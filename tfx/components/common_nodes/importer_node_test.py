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
"""Tests for tfx.components.model_validator.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.common_nodes import importer_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import standard_artifacts


class ImporterNodeTest(tf.test.TestCase):

  def testImporterDefinition(self):
    impt = importer_node.ImporterNode(
        instance_name='my_importer',
        source_uri='m/y/u/r/i',
        artifact_type=standard_artifacts.Examples)
    self.assertDictEqual(
        impt.exec_properties, {
            importer_node.SOURCE_URI_KEY: 'm/y/u/r/i',
            importer_node.REIMPORT_OPTION_KEY: False
        })
    self.assertEmpty(impt.inputs.get_all())
    self.assertEqual(
        impt.outputs.get_all()[importer_node.IMPORT_RESULT_KEY].type_name,
        standard_artifacts.Examples.TYPE_NAME)


class ImporterDriverTest(tf.test.TestCase):

  def setUp(self):
    super(ImporterDriverTest, self).setUp()
    self.connection_config = metadata_store_pb2.ConnectionConfig()
    self.connection_config.sqlite.SetInParent()
    self.artifact_type = 'Examples'
    self.output_dict = {
        importer_node.IMPORT_RESULT_KEY:
            types.Channel(type_name=self.artifact_type)
    }
    self.source_uri = 'm/y/u/r/i'
    self.existing_artifact = types.Artifact(type_name=self.artifact_type)
    self.existing_artifact.uri = self.source_uri
    self.component_info = data_types.ComponentInfo(
        component_type='c_type', component_id='c_id')
    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name='p_name', pipeline_root='p_root', run_id='run_id')
    self.driver_args = data_types.DriverArgs(enable_cache=True)

  def _callImporterDriver(self, reimport: bool):
    with metadata.Metadata(connection_config=self.connection_config) as m:
      m.publish_artifacts([self.existing_artifact])
      driver = importer_node.ImporterDriver(metadata_handler=m)
      execution_result = driver.pre_execution(
          component_info=self.component_info,
          pipeline_info=self.pipeline_info,
          driver_args=self.driver_args,
          input_dict={},
          output_dict=self.output_dict,
          exec_properties={
              importer_node.SOURCE_URI_KEY: self.source_uri,
              importer_node.REIMPORT_OPTION_KEY: reimport
          })
      self.assertFalse(execution_result.use_cached_results)
      self.assertEmpty(execution_result.input_dict)
      self.assertEqual(
          execution_result.output_dict[importer_node.IMPORT_RESULT_KEY][0].uri,
          self.source_uri)
      self.assertEqual(
          execution_result.output_dict[importer_node.IMPORT_RESULT_KEY][0].id,
          2 if reimport else 1)

  def testImportArtifact(self):
    self._callImporterDriver(reimport=True)

  def testReuseArtifact(self):
    self._callImporterDriver(reimport=False)


if __name__ == '__main__':
  tf.test.main()
