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
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils


class ImporterNodeTest(tf.test.TestCase):

  def testImporterDefinitionWithSingleUri(self):
    impt = importer_node.ImporterNode(
        instance_name='my_importer',
        source_uri='m/y/u/r/i',
        artifact_type=standard_artifacts.Examples)
    self.assertDictEqual(
        impt.exec_properties, {
            importer_node.SOURCE_URI_KEY: ['m/y/u/r/i'],
            importer_node.REIMPORT_OPTION_KEY: False,
            importer_node.SPLIT_KEY: [''],
        })
    self.assertEmpty(impt.inputs.get_all())
    self.assertEqual(
        impt.outputs.get_all()[importer_node.IMPORT_RESULT_KEY].type_name,
        standard_artifacts.Examples.TYPE_NAME)

  def testImporterDefinitionWithMultipleUris(self):
    impt = importer_node.ImporterNode(
        instance_name='my_importer',
        source_uri=['m/y/u/r/i/1', 'm/y/u/r/i/2'],
        artifact_type=standard_artifacts.Examples,
        split=['train', 'eval'])
    self.assertDictEqual(
        impt.exec_properties, {
            importer_node.SOURCE_URI_KEY: ['m/y/u/r/i/1', 'm/y/u/r/i/2'],
            importer_node.REIMPORT_OPTION_KEY: False,
            importer_node.SPLIT_KEY: ['train', 'eval'],
        })
    self.assertEqual([
        artifact_utils.decode_split_names(s.split_names)[0]
        for s in impt.outputs.get_all()[importer_node.IMPORT_RESULT_KEY].get()
    ], ['train', 'eval'])

  def testImporterDefinitionWithMultipleUrisBadSplitSpecification(self):
    with self.assertRaises(ValueError):
      _ = importer_node.ImporterNode(
          instance_name='my_importer',
          source_uri=['m/y/u/r/i/1', 'm/y/u/r/i/2'],
          artifact_type=standard_artifacts.Examples,
      )

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
    self.source_uri = ['m/y/u/r/i/1', 'm/y/u/r/i/2']
    self.split = ['train', 'eval']

    self.existing_artifacts = []
    for uri, split in zip(self.source_uri, self.split):
      existing_artifact = standard_artifacts.Examples()
      existing_artifact.uri = uri
      existing_artifact.split_names = artifact_utils.encode_split_names([split])
      self.existing_artifacts.append(existing_artifact)

    self.component_info = data_types.ComponentInfo(
        component_type='c_type', component_id='c_id')
    self.pipeline_info = data_types.PipelineInfo(
        pipeline_name='p_name', pipeline_root='p_root', run_id='run_id')
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
              importer_node.SPLIT_KEY: self.split,
          })
      self.assertFalse(execution_result.use_cached_results)
      self.assertEmpty(execution_result.input_dict)
      self.assertEqual(
          execution_result.output_dict[importer_node.IMPORT_RESULT_KEY][0].uri,
          self.source_uri[0])
      self.assertEqual(
          execution_result.output_dict[importer_node.IMPORT_RESULT_KEY][0].id,
          3 if reimport else 1)

      self.assertNotEmpty(
          self.output_dict[importer_node.IMPORT_RESULT_KEY].get())

      results = self.output_dict[importer_node.IMPORT_RESULT_KEY].get()
      for res, uri, split in zip(results, self.source_uri, self.split):
        self.assertEqual(res.uri, uri)
        self.assertEqual(
            artifact_utils.decode_split_names(res.split_names)[0], split)

  def testImportArtifact(self):
    self._callImporterDriver(reimport=True)

  def testReuseArtifact(self):
    self._callImporterDriver(reimport=False)


if __name__ == '__main__':
  tf.test.main()
