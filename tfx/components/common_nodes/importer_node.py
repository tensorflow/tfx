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
"""TFX Importer definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
from typing import Any, Dict, List, Optional, Text, Type

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.types import artifact
from tfx.types import node_common

# Constant to access importer importing result from importer output dict.
IMPORT_RESULT_KEY = 'result'
# Constant to access source uri from importer exec_properties dict.
SOURCE_URI_KEY = 'source_uri'
# Constant to access re-import option from importer exec_properties dict.
REIMPORT_OPTION_KEY = 'reimport'


class ImporterDriver(base_driver.BaseDriver):
  """Driver for Importer."""

  def _import_artifacts(
      self,
      source_uri: Text,
      reimport: bool,
      destination_channel: types.Channel,
  ) -> List[types.Artifact]:
    """Imports external resource in MLMD."""
    absl.logging.info('Processing source uri: %s' % source_uri)

    previous_artifacts = self._metadata_handler.get_artifacts_by_uri(source_uri)
    result = types.Artifact(type_name=destination_channel.type_name)
    result.uri = source_uri

    # If any registered artifact with the same uri also has the same
    # fingerprint and user does not ask for re-import, just reuse the latest.
    # Otherwise, register the external resource into MLMD using the type info
    # in the destination channel.
    if bool(previous_artifacts) and not reimport:
      absl.logging.info('Reusing existing artifact')
      result.set_artifact(max(previous_artifacts, key=lambda m: m.id))
    else:
      [registered_artifact] = self._metadata_handler.publish_artifacts([result])
      absl.logging.info('Registered new artifact: %s' % registered_artifact)
      result.set_artifact(registered_artifact)

    return [result]

  def pre_execution(
      self,
      input_dict: Dict[Text, types.Channel],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    output_artifacts = {
        IMPORT_RESULT_KEY:
            self._import_artifacts(
                source_uri=exec_properties[SOURCE_URI_KEY],
                destination_channel=output_dict[IMPORT_RESULT_KEY],
                reimport=exec_properties[REIMPORT_OPTION_KEY])
    }
    return data_types.ExecutionDecision(
        input_dict={},
        output_dict=output_artifacts,
        exec_properties={},
        execution_id=self._register_execution(
            exec_properties={},
            pipeline_info=pipeline_info,
            component_info=component_info),
        use_cached_results=False)


class ImporterNode(base_node.BaseNode):
  """Definition for TFX ImporterNode.

  ImporterNode is a special TFX node which registers an external resource into
  MLMD
  so that downstream nodes can use the registered artifact as input.

  Here is an example to use ImporterNode:

  ...
  importer = ImporterNode(
      instance_name='import_schema',
      source_uri='uri/to/schema'
      artifact_type=standard_artifact.Schema,
      reimport=False)
  schema_gen = SchemaGen(
      fixed_schema=importer.outputs['result'],
      examples=...)
  ...

  Attributes:
    _source_uri: the source uri to import.
    _reimport: whether or not to re-import the URI even if it already exists in
      MLMD.
  """

  DRIVER_CLASS = ImporterDriver

  def __init__(self,
               instance_name: Text,
               source_uri: Text,
               artifact_type: Type[artifact.Artifact],
               reimport: Optional[bool] = False):
    """Init function for ImporterNode.

    Args:
      instance_name: the name of the ImporterNode instance.
      source_uri: the URI to the resource that needs to be registered.
      artifact_type: the type of the artifact to import.
      reimport: whether or not to re-import as a new artifact if the URI has
        been imported in before.
    """
    self._output_dict = {
        'result': types.Channel(
            type=artifact_type, artifacts=[artifact_type()])
    }
    self._source_uri = source_uri
    self._reimport = reimport
    super(ImporterNode, self).__init__(instance_name=instance_name)

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:
    return node_common._PropertyDictWrapper({})  # pylint: disable=protected-access

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:
    return node_common._PropertyDictWrapper(self._output_dict)  # pylint: disable=protected-access

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return {
        SOURCE_URI_KEY: self._source_uri,
        REIMPORT_OPTION_KEY: self._reimport
    }
