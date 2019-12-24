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
import tensorflow as tf

from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import channel_utils


def _generate_output_uri(artifact: types.Artifact, base_output_dir: Text,
                         name: Text, execution_id: int) -> Text:
  """Generate uri for output artifact."""

  # Generates outputs uri based on execution id and optional split.
  # Last empty string forces this be to a directory.
  uri = os.path.join(base_output_dir, name, str(execution_id), artifact.split,
                     '')
  if tf.io.gfile.exists(uri):
    msg = 'Output artifact uri %s already exists' % uri
    absl.logging.error(msg)
    raise RuntimeError(msg)
  else:
    # TODO(zhitaoli): Consider refactoring this out into something
    # which can handle permission bits.
    absl.logging.debug('Creating output artifact uri %s as directory', uri)
    tf.io.gfile.makedirs(uri)

  return uri


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
        if not tf.io.gfile.exists(artifact.uri):
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
      Final execution properties that will be used in execution.

    Raises:
      RuntimeError: for Channels that do not contain any artifact. This will be
        reverted once we support Channel-based input resolution.
      ValueError: if in interactive mode, the given input channels have not been
        resolved.
    """
    result = {}
    for name, input_channel in input_dict.items():
      artifacts = list(input_channel.get())
      if driver_args.interactive_resolution:
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
        continue
      # TODO(ruoyu): Remove once channel-based input resolution is supported.
      if not artifacts:
        raise RuntimeError('Channel-based input resolution is not supported.')
      result[name] = self._metadata_handler.search_artifacts(
          artifacts[0].name, pipeline_info.pipeline_name, pipeline_info.run_id,
          artifacts[0].producer_component)
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
      output_dict: Dict[Text, types.Channel],
      execution_id: int,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Prepare output artifacts by assigning uris to each artifact."""
    result = channel_utils.unwrap_channel_dict(output_dict)
    base_output_dir = os.path.join(pipeline_info.pipeline_root,
                                   component_info.component_id)
    for name, output_list in result.items():
      for artifact in output_list:
        artifact.uri = _generate_output_uri(artifact, base_output_dir, name,
                                            execution_id)
    return result

  def _fetch_cached_artifacts(
      self, output_dict: Dict[Text, types.Channel],
      cached_execution_id: int) -> Dict[Text, List[types.Artifact]]:
    """Fetch cached output artifacts."""
    output_artifacts_dict = channel_utils.unwrap_channel_dict(output_dict)
    return self._metadata_handler.fetch_previous_result_artifacts(
        output_artifacts_dict, cached_execution_id)

  def _register_execution(self, exec_properties: Dict[Text, Any],
                          pipeline_info: data_types.PipelineInfo,
                          component_info: data_types.ComponentInfo):
    """Register the upcoming execution in MLMD.

    Args:
      exec_properties: Dict of other execution properties.
      pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
        related properties including pipeline_name, pipeline_root and run_id
      component_info: An instance of data_types.ComponentInfo, holding component
        related properties including component_type and component_id.

    Returns:
      the id of the upcoming execution
    """
    run_context_id = self._metadata_handler.register_run_context_if_not_exists(
        pipeline_info)
    execution_id = self._metadata_handler.register_execution(
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info,
        run_context_id=run_context_id)
    absl.logging.debug('Execution id of the upcoming component execution is %s',
                       execution_id)
    return execution_id

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
    input_artifacts = self.resolve_input_artifacts(input_dict, exec_properties,
                                                   driver_args, pipeline_info)
    self.verify_input_artifacts(artifacts_dict=input_artifacts)
    absl.logging.debug('Resolved input artifacts are: %s', input_artifacts)
    # Step 2. Register execution in metadata.
    execution_id = self._register_execution(
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info)
    output_artifacts = {}
    use_cached_results = False

    if driver_args.enable_cache:
      # TODO(b/136031301): Combine Step 3 and Step 4b after finish migration to
      # go/tfx-oss-artifact-passing.
      # Step 3. Decide whether a new execution is needed.
      cached_execution_id = self._metadata_handler.previous_execution(
          input_artifacts=input_artifacts,
          exec_properties=exec_properties,
          pipeline_info=pipeline_info,
          component_info=component_info)
      if cached_execution_id:
        absl.logging.debug('Found cached_execution: %s', cached_execution_id)
        # Step 4b. New execution not needed. Fetch cached output artifacts.
        try:
          output_artifacts = self._fetch_cached_artifacts(
              output_dict=output_dict, cached_execution_id=cached_execution_id)
          absl.logging.debug('Cached output artifacts are: %s',
                             output_artifacts)
          use_cached_results = True
        except RuntimeError:
          absl.logging.warning(
              'Error when trying to get cached output artifacts')
          use_cached_results = False
    if not use_cached_results:
      absl.logging.debug('Cached results not found, move on to new execution')
      # Step 4a. New execution is needed. Prepare output artifacts.
      output_artifacts = self._prepare_output_artifacts(
          output_dict=output_dict,
          execution_id=execution_id,
          pipeline_info=pipeline_info,
          component_info=component_info)
      absl.logging.debug(
          'Output artifacts skeleton for the upcoming execution are: %s',
          output_artifacts)
      exec_properties = self.resolve_exec_properties(exec_properties,
                                                     pipeline_info,
                                                     component_info)
      absl.logging.debug(
          'Execution properties for the upcoming execution are: %s',
          exec_properties)

    return data_types.ExecutionDecision(input_artifacts, output_artifacts,
                                        exec_properties, execution_id,
                                        use_cached_results)
