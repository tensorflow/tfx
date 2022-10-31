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

from typing import Any, Dict, List, Optional, Type, Union

import absl
from tfx import types
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import channel_utils
from tfx.utils import doc_controls

from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2

# Constant to access Importer importing result from Importer output dict.
IMPORT_RESULT_KEY = 'result'
# Constant to access output key from Importer exec_properties dict.
OUTPUT_KEY_KEY = 'output_key'
# Constant to access artifact uri from Importer exec_properties dict.
SOURCE_URI_KEY = 'artifact_uri'
# Constant to access re-import option from Importer exec_properties dict.
REIMPORT_OPTION_KEY = 'reimport'


def _set_artifact_properties(artifact: types.Artifact,
                             properties: Optional[Dict[str, Any]],
                             custom_properties: Optional[Dict[str, Any]]):
  """Sets properties and custom_properties to the given artifact."""
  if properties is not None:
    for key, value in properties.items():
      setattr(artifact, key, value)
  if custom_properties is not None:
    for key, value in custom_properties.items():
      if isinstance(value, int):
        artifact.set_int_custom_property(key, value)
      elif isinstance(value, (str, bytes)):
        artifact.set_string_custom_property(key, value)
      else:
        raise NotImplementedError(
            f'Unexpected custom_property value type:{type(value)}')


def _prepare_artifact(
    metadata_handler: metadata.Metadata,
    uri: str,
    properties: Dict[str, Any],
    custom_properties: Dict[str, Any],
    reimport: bool, output_artifact_class: Type[types.Artifact],
    mlmd_artifact_type: Optional[metadata_store_pb2.ArtifactType]
) -> types.Artifact:
  """Prepares the Importer's output artifact.

  If there is already an artifact in MLMD with the same URI, properties /
  custom properties, and type, that artifact will be reused unless the
  `reimport` argument is set to True.

  Args:
    metadata_handler: The handler of MLMD.
    uri: The uri of the artifact.
    properties: The properties of the artifact, given as a dictionary from
      string keys to integer / string values. Must conform to the declared
      properties of the destination channel's output type.
    custom_properties: The custom properties of the artifact, given as a
      dictionary from string keys to integer / string values.
    reimport: If set to True, will register a new artifact even if it already
      exists in the database.
    output_artifact_class: The class of the output artifact.
    mlmd_artifact_type: The MLMD artifact type of the Artifact to be created.

  Returns:
    An Artifact object representing the imported artifact.
  """
  absl.logging.info(
      'Processing source uri: %s, properties: %s, custom_properties: %s' %
      (uri, properties, custom_properties))

  # Check types of custom properties.
  for key, value in custom_properties.items():
    if not isinstance(value, (int, str, bytes)):
      raise ValueError(
          ('Custom property value for key %r must be a string or integer '
           '(got %r instead)') % (key, value))

  unfiltered_previous_artifacts = metadata_handler.get_artifacts_by_uri(uri)
  new_artifact_type = False
  if mlmd_artifact_type and not mlmd_artifact_type.id:
    try:
      mlmd_artifact_type = metadata_handler.store.get_artifact_type(
          mlmd_artifact_type.name)
    except errors.NotFoundError:
      # Artifact type is not registered, so it must be new.
      new_artifact_type = True

  result = output_artifact_class(mlmd_artifact_type)
  result.uri = uri
  result.is_external = True
  _set_artifact_properties(result, properties, custom_properties)

  # Only consider previous artifacts as candidates to reuse if:
  #  * reimport is False
  #  * the given artifact type is recognized by MLMD
  #  * the type and properties match the imported artifact
  if not reimport and not new_artifact_type:
    previous_artifacts = []
    for candidate_mlmd_artifact in unfiltered_previous_artifacts:
      if mlmd_artifact_type and candidate_mlmd_artifact.type_id != mlmd_artifact_type.id:
        # If mlmd_artifact_type is defined, don't reuse existing artifacts if
        # they don't match the given type.
        continue
      is_candidate = True
      candidate_artifact = output_artifact_class(mlmd_artifact_type)
      candidate_artifact.set_mlmd_artifact(candidate_mlmd_artifact)
      for key, value in properties.items():
        if getattr(candidate_artifact, key) != value:
          is_candidate = False
          break
      for key, value in custom_properties.items():
        if isinstance(value, int):
          if candidate_artifact.get_int_custom_property(key) != value:
            is_candidate = False
            break
        elif isinstance(value, (str, bytes)):
          if candidate_artifact.get_string_custom_property(key) != value:
            is_candidate = False
            break
      if is_candidate:
        previous_artifacts.append(candidate_mlmd_artifact)
    # If a registered artifact has the same uri and properties and the user does
    # not explicitly ask for reimport, reuse that artifact.
    if previous_artifacts:
      absl.logging.info('Reusing existing artifact')
      result.set_mlmd_artifact(
          max(previous_artifacts, key=lambda m: m.create_time_since_epoch))

  return result


def generate_output_dict(
    metadata_handler: metadata.Metadata,
    uri: str,
    properties: Dict[str, Any],
    custom_properties: Dict[str, Any],
    reimport: bool,
    output_artifact_class: Type[types.Artifact],
    mlmd_artifact_type: Optional[metadata_store_pb2.ArtifactType] = None,
    output_key: Optional[str] = None,
) -> Dict[str, List[types.Artifact]]:
  """Generates Importer's output dict.

  If there is already an artifact in MLMD with the same URI and properties /
  custom properties, that artifact will be reused unless the `reimport`
  argument is set to True.

  Args:
    metadata_handler: The handler of MLMD.
    uri: The uri of the artifact.
    properties: The properties of the artifact, given as a dictionary from
      string keys to integer / string values. Must conform to the declared
      properties of the destination channel's output type.
    custom_properties: The custom properties of the artifact, given as a
      dictionary from string keys to integer / string values.
    reimport: If set to True, will register a new artifact even if it already
      exists in the database.
    output_artifact_class: The class of the output artifact.
    mlmd_artifact_type: The MLMD artifact type of the Artifact to be created.
    output_key: The key to use for the imported artifact in the Importer's
      output dictionary. Defaults to 'result'.

  Returns:
    A dictionary with the only key `output_key` whose value is the Artifact.
  """
  output_key = output_key or IMPORT_RESULT_KEY
  return {
      output_key: [
          _prepare_artifact(
              metadata_handler,
              uri=uri,
              properties=properties,
              custom_properties=custom_properties,
              output_artifact_class=output_artifact_class,
              mlmd_artifact_type=mlmd_artifact_type,
              reimport=reimport)
      ]
  }


class ImporterDriver(base_driver.BaseDriver):
  """Driver for Importer."""

  def pre_execution(
      self,
      input_dict: Dict[str, types.BaseChannel],
      output_dict: Dict[str, types.Channel],
      exec_properties: Dict[str, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    # Registers contexts and execution.
    contexts = self._metadata_handler.register_pipeline_contexts_if_not_exists(
        pipeline_info)
    execution = self._metadata_handler.register_execution(
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info,
        contexts=contexts)
    # Create imported artifacts.
    output_key = exec_properties[OUTPUT_KEY_KEY]
    output_channel = output_dict[output_key]
    output_artifacts = generate_output_dict(
        self._metadata_handler,
        uri=exec_properties[SOURCE_URI_KEY],
        properties=output_channel.additional_properties,
        custom_properties=output_channel.additional_custom_properties,
        reimport=exec_properties[REIMPORT_OPTION_KEY],
        output_artifact_class=output_channel.type,
        output_key=output_key)

    # Update execution with imported artifacts.
    self._metadata_handler.update_execution(
        execution=execution,
        component_info=component_info,
        output_artifacts=output_artifacts,
        execution_state=metadata.EXECUTION_STATE_CACHED,
        contexts=contexts)

    output_dict[output_key] = channel_utils.as_channel(
        output_artifacts[output_key])

    return data_types.ExecutionDecision(
        input_dict={},
        output_dict=output_artifacts,
        exec_properties=exec_properties,
        execution_id=execution.id,
        use_cached_results=False)


class Importer(base_node.BaseNode):
  """Definition for TFX Importer.

  The Importer is a special TFX node which registers an external resource into
  MLMD so that downstream nodes can use the registered artifact as an input.

  Here is an example to use the Importer:

  ```
  importer = Importer(
      source_uri='uri/to/schema',
      artifact_type=standard_artifacts.Schema,
      reimport=False).with_id('import_schema')
  schema_gen = SchemaGen(
      fixed_schema=importer.outputs['result'],
      examples=...)
  ```
  """

  def __init__(self,
               source_uri: str,
               artifact_type: Type[types.Artifact],
               reimport: Optional[bool] = False,
               properties: Optional[Dict[str, Union[str, int]]] = None,
               custom_properties: Optional[Dict[str, Union[str, int]]] = None,
               output_key: Optional[str] = None):
    """Init function for the Importer.

    Args:
      source_uri: the URI of the resource that needs to be registered.
      artifact_type: the type of the artifact to import.
      reimport: whether or not to re-import as a new artifact if the URI has
        been imported in before.
      properties: Dictionary of properties for the imported Artifact. These
        properties should be ones declared for the given artifact_type (see the
        PROPERTIES attribute of the definition of the type for details).
      custom_properties: Dictionary of custom properties for the imported
        Artifact. These properties should be of type Text or int.
      output_key: The key to use for the imported artifact in the Importer's
        output dictionary. Defaults to 'result'.
    """
    self._source_uri = source_uri
    self._reimport = reimport
    self._output_key = output_key or IMPORT_RESULT_KEY

    artifact = artifact_type()
    artifact.is_external = True
    _set_artifact_properties(artifact, properties, custom_properties)

    output_channel = types.OutputChannel(
        artifact_type,
        producer_component=self,
        output_key=self._output_key,
        additional_properties=properties,
        additional_custom_properties=custom_properties)

    # TODO(b/161490287): remove static artifacts.
    output_channel.set_artifacts([artifact])
    self._output_dict = {self._output_key: output_channel}

    super().__init__(driver_class=ImporterDriver)

  @property
  @doc_controls.do_not_generate_docs
  def inputs(self) -> Dict[str, Any]:
    return {}

  @property
  def outputs(self) -> Dict[str, Any]:
    """Output Channel dict that contains imported artifacts."""
    return self._output_dict

  @property
  @doc_controls.do_not_generate_docs
  def exec_properties(self) -> Dict[str, Any]:
    return {
        SOURCE_URI_KEY: self._source_uri,
        REIMPORT_OPTION_KEY: int(self._reimport),
        OUTPUT_KEY_KEY: self._output_key,
    }
