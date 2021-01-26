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
"""Abstract TFX driver class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

import absl

from tfx import types
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata


def _generate_output_uri(base_output_dir: Text,
                         name: Text,
                         execution_id: int,
                         is_single_artifact: bool = True,
                         index: int = 0) -> Text:
  """Generate uri for output artifact."""
  if is_single_artifact:
    # TODO(b/145680633): Consider differentiating different types of uris.
    return os.path.join(base_output_dir, name, str(execution_id))

  return os.path.join(base_output_dir, name, str(execution_id), str(index))


def _prepare_output_paths(artifact: types.Artifact):
  """Create output directories for output artifact."""
  if fileio.exists(artifact.uri):
    msg = 'Output artifact uri %s already exists' % artifact.uri
    absl.logging.warning(msg)
    # TODO(b/158689199): We currently simply return as a short-term workaround
    # to unblock execution retires. A comprehensive solution to guarantee
    # idempotent executions is needed.
    return

  # TODO(b/147242148): Introduce principled artifact structure (directory
  # or file) definition.
  if isinstance(artifact, types.ValueArtifact):
    artifact_dir = os.path.dirname(artifact.uri)
  else:
    artifact_dir = artifact.uri

  # TODO(zhitaoli): Consider refactoring this out into something
  # which can handle permission bits.
  absl.logging.debug('Creating output artifact uri %s as directory',
                     artifact_dir)
  fileio.makedirs(artifact_dir)


class BaseDriver(object):
  """BaseDriver is the base class of all custom drivers.

  This can also be used as the default driver of a component if no custom logic
  is needed.

  Attributes:
    _metadata_handler: An instance of Metadata.
  """

  def __init__(self, metadata_handler: metadata.Metadata):
    self._metadata_handler = metadata_handler

  def verify_input_artifacts(
      self, artifacts_dict: Dict[Text, List[types.Artifact]]) -> None:
    """Verify that all artifacts have existing uri.

    Args:
      artifacts_dict: key -> types.Artifact for inputs.

    Raises:
      RuntimeError: if any input as an empty or non-existing uri.
    """
    for single_artifacts_list in artifacts_dict.values():
      for artifact in single_artifacts_list:
        if not artifact.uri:
          raise RuntimeError('Artifact %s does not have uri' % artifact)
        if not fileio.exists(artifact.uri):
          raise RuntimeError('Artifact uri %s is missing' % artifact.uri)

  def _log_properties(self, input_dict: Dict[Text, List[types.Artifact]],
                      output_dict: Dict[Text, List[types.Artifact]],
                      exec_properties: Dict[Text, Any]):
    """Log inputs, outputs, and executor properties in a standard format."""
    absl.logging.debug('Starting %s driver.', self.__class__.__name__)
    absl.logging.debug('Inputs for %s are: %s', self.__class__.__name__,
                       input_dict)
    absl.logging.debug('Execution properties for %s are: %s',
                       self.__class__.__name__, exec_properties)
    absl.logging.debug('Outputs for %s are: %s', self.__class__.__name__,
                       output_dict)

  def resolve_input_artifacts(
      self,
      input_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Resolve input artifacts from metadata.

    Subclasses might override this function for customized artifact properties
    resolution logic. However please note that this function is supposed to be
    called in normal cases (except head of the pipeline) since it handles
    artifact info passing from upstream components.

    Args:
      input_dict: key -> Channel mapping for inputs generated in logical
        pipeline.
      exec_properties: Dict of other execution properties, e.g., configs.
      driver_args: An instance of data_types.DriverArgs with driver
        configuration properties.
      pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
        related properties including component_type and component_id.

    Returns:
      Final artifacts that will be used in execution.

    Raises:
      ValueError: if in interactive mode, the given input channels have not been
        resolved.
    """
    result = {}
    for name, input_channel in input_dict.items():
      if driver_args.interactive_resolution:
        artifacts = list(input_channel.get())
        for artifact in artifacts:
          # Note: when not initialized, artifact.uri is '' and artifact.id is 0.
          if not artifact.uri or not artifact.id:
            raise ValueError((
                'Unresolved input channel %r for input %r was passed in '
                'interactive mode. When running in interactive mode, upstream '
                'components must first be run with '
                '`interactive_context.run(component)` before their outputs can '
                'be used in downstream components.') % (artifact, name))
        result[name] = artifacts
      else:
        result[name] = self._metadata_handler.search_artifacts(
            artifact_name=input_channel.output_key,
            pipeline_info=pipeline_info,
            producer_component_id=input_channel.producer_component_id)
        # TODO(ccy): add this code path to interactive resolution.
        for artifact in result[name]:
          if isinstance(artifact, types.ValueArtifact):
            # Resolve the content of file into value field for value artifacts.
            _ = artifact.read()
    return result

  def resolve_exec_properties(
      self,
      exec_properties: Dict[Text, Any],
      pipeline_info: data_types.PipelineInfo,  # pylint: disable=unused-argument
      component_info: data_types.ComponentInfo,  # pylint: disable=unused-argument
  ) -> Dict[Text, Any]:
    """Resolve execution properties.

    Subclasses might override this function for customized execution properties
    resolution logic.

    Args:
      exec_properties: Original execution properties passed in.
      pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
        related properties including pipeline_name, pipeline_root and run_id
      component_info: An instance of data_types.ComponentInfo, holding component
        related properties including component_type and component_id.

    Returns:
      Final execution properties that will be used in execution.
    """
    return exec_properties

  def _prepare_output_artifacts(
      self,
      input_artifacts: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      execution_id: int,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Prepare output artifacts by assigning uris to each artifact."""
    del exec_properties

    base_output_dir = os.path.join(pipeline_info.pipeline_root,
                                   component_info.component_id)

    result = {}
    for name, channel in output_dict.items():
      if channel.matching_channel_name:
        # Decides the artifact count for output Channel at runtime based on the
        # artifact count in specified input Channel.
        count = len(input_artifacts[channel.matching_channel_name])
        output_list = [channel.type() for _ in range(count)]
      else:
        output_list = [channel.type()]

      is_single_artifact = len(output_list) == 1
      for i, artifact in enumerate(output_list):
        artifact.name = name
        artifact.producer_component = component_info.component_id
        artifact.uri = _generate_output_uri(base_output_dir, name, execution_id,
                                            is_single_artifact, i)
        # TODO(b/147242148): Introduce principled artifact structure (directory
        # or file) definition.
        if isinstance(artifact, types.ValueArtifact):
          artifact.uri = os.path.join(artifact.uri, 'value')
        _prepare_output_paths(artifact)

      result[name] = output_list

    return result

  def pre_execution(
      self,
      input_dict: Dict[Text, types.Channel],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    """Handle pre-execution logic.

    There are four steps:
      1. Fetches input artifacts from metadata and checks whether uri exists.
      2. Registers execution.
      3. Decides whether a new execution is needed.
      4a. If (3), prepare output artifacts.
      4b. If not (3), fetch cached output artifacts.

    Args:
      input_dict: key -> Channel for inputs.
      output_dict: key -> Channel for outputs. Uris of the outputs are not
        assigned.
      exec_properties: Dict of other execution properties.
      driver_args: An instance of data_types.DriverArgs class.
      pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
        related properties including pipeline_name, pipeline_root and run_id
      component_info: An instance of data_types.ComponentInfo, holding component
        related properties including component_type and component_id.

    Returns:
      data_types.ExecutionDecision object.

    Raises:
      RuntimeError: if any input as an empty uri.
    """
    # Step 1. Fetch inputs from metadata.
    exec_properties = self.resolve_exec_properties(exec_properties,
                                                   pipeline_info,
                                                   component_info)
    input_artifacts = self.resolve_input_artifacts(input_dict, exec_properties,
                                                   driver_args, pipeline_info)
    self.verify_input_artifacts(artifacts_dict=input_artifacts)
    absl.logging.debug('Resolved input artifacts are: %s', input_artifacts)
    # Step 2. Register execution in metadata.
    contexts = self._metadata_handler.register_pipeline_contexts_if_not_exists(
        pipeline_info)
    execution = self._metadata_handler.register_execution(
        input_artifacts=input_artifacts,
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info,
        contexts=contexts)
    use_cached_results = False
    output_artifacts = None

    if driver_args.enable_cache:
      # Step 3. Decide whether a new execution is needed.
      output_artifacts = self._metadata_handler.get_cached_outputs(
          input_artifacts=input_artifacts,
          exec_properties=exec_properties,
          pipeline_info=pipeline_info,
          component_info=component_info)
    if output_artifacts is not None:
      # If cache should be used, updates execution to reflect that. Note that
      # with this update, publisher should / will be skipped.
      self._metadata_handler.update_execution(
          execution=execution,
          component_info=component_info,
          output_artifacts=output_artifacts,
          execution_state=metadata.EXECUTION_STATE_CACHED,
          contexts=contexts)
      use_cached_results = True
    else:
      absl.logging.debug('Cached results not found, move on to new execution')
      # Step 4a. New execution is needed. Prepare output artifacts.
      output_artifacts = self._prepare_output_artifacts(
          input_artifacts=input_artifacts,
          output_dict=output_dict,
          exec_properties=exec_properties,
          execution_id=execution.id,
          pipeline_info=pipeline_info,
          component_info=component_info)
      absl.logging.debug(
          'Output artifacts skeleton for the upcoming execution are: %s',
          output_artifacts)
      # Updates the execution to reflect refreshed output artifacts and
      # execution properties.
      self._metadata_handler.update_execution(
          execution=execution,
          component_info=component_info,
          output_artifacts=output_artifacts,
          exec_properties=exec_properties,
          contexts=contexts)
      absl.logging.debug(
          'Execution properties for the upcoming execution are: %s',
          exec_properties)

    # For interactive execution, update the output channel contents.
    # TODO(b/161490287): figure out the long-term behavior of Channel artifacts
    # with respect to interactive and non-interactive execution.
    if driver_args.interactive_resolution:
      for key, artifact_list in output_artifacts.items():
        channel = output_dict[key]
        channel._artifacts = artifact_list  # pylint: disable=protected-access

    return data_types.ExecutionDecision(input_artifacts, output_artifacts,
                                        exec_properties, execution.id,
                                        use_cached_results)
