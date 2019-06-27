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
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text

from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.utils import channel
from tfx.utils import types


class BaseDriver(object):
  """BaseDriver is the base class of all custom drivers.

  This can also be used as the default driver of a component if no custom logic
  is needed.

  Attributes:
    _metadata_handler: An instance of Metadata.
  """

  # TODO(b/131703697): Remove the need for constructor to make driver stateless.
  def __init__(self, metadata_handler: metadata.Metadata):
    self._metadata_handler = metadata_handler

  def _log_properties(self, input_dict: Dict[Text, List[types.TfxArtifact]],
                      output_dict: Dict[Text, List[types.TfxArtifact]],
                      exec_properties: Dict[Text, Any]):
    """Log inputs, outputs, and executor properties in a standard format."""
    tf.logging.info('Starting %s driver.', self.__class__.__name__)
    tf.logging.info('Inputs for {} is: {}'.format(self.__class__.__name__,
                                                  input_dict))
    tf.logging.info('Execution properties for {} is: {}'.format(
        self.__class__.__name__, exec_properties))
    tf.logging.info('Outputs for {} is: {}'.format(self.__class__.__name__,
                                                   output_dict))

  def _get_output_from_previous_run(
      self,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      output_dict: Dict[Text, List[types.TfxArtifact]],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
  ) -> Optional[Dict[Text, List[types.TfxArtifact]]]:
    """Returns outputs from previous identical execution if found."""
    previous_execution_id = self._metadata_handler.previous_run(
        type_name=driver_args.worker_name,
        input_dict=input_dict,
        exec_properties=exec_properties)
    if previous_execution_id:
      final_output = self._metadata_handler.fetch_previous_result_artifacts(
          output_dict, previous_execution_id)
      for output_list in final_output.values():
        for single_output in output_list:
          if not single_output.uri or not tf.gfile.Exists(single_output.uri):
            tf.logging.warning(
                'URI of cached artifact %s does not exist, forcing new execution',
                single_output)
            return None
      tf.logging.info(
          'Reusing previous execution {} output artifacts {}'.format(
              previous_execution_id, final_output))
      return final_output
    else:
      return None

  def _verify_artifacts(self,
                        artifacts_dict: Dict[Text, List[types.TfxArtifact]]
                       ) -> None:
    """Verify that all artifacts have existing uri.

    Args:
      artifacts_dict: key -> TfxArtifact for inputs.

    Raises:
      RuntimeError: if any input as an empty or non-existing uri.
    """
    for single_artifacts_list in artifacts_dict.values():
      for artifact in single_artifacts_list:
        if not artifact.uri:
          raise RuntimeError('Artifact {} does not have uri'.format(artifact))
        if not tf.gfile.Exists(os.path.dirname(artifact.uri)):
          raise RuntimeError('Artifact uri {} is missing'.format(artifact.uri))

  def _generate_output_uri(self, artifact: types.TfxArtifact,
                           base_output_dir: Text, name: Text,
                           execution_id: int) -> Text:
    """Generate uri for output artifact."""

    # Generates outputs uri based on execution id and optional split.
    # Last empty string forces this be to a directory.
    uri = os.path.join(base_output_dir, name, str(execution_id), artifact.split,
                       '')
    if tf.gfile.Exists(uri):
      msg = 'Output artifact uri {} already exists'.format(uri)
      tf.logging.error(msg)
      raise RuntimeError(msg)
    else:
      # TODO(zhitaoli): Consider refactoring this out into something
      # which can handle permission bits.
      tf.logging.info('Creating output artifact uri %s as directory', uri)
      tf.gfile.MakeDirs(uri)

    return uri

  def _default_caching_handling(
      self,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      output_dict: Dict[Text, List[types.TfxArtifact]],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
  ) -> data_types.ExecutionDecision:
    """Check cache for desired and applicable identical execution."""
    enable_cache = driver_args.enable_cache
    base_output_dir = driver_args.base_output_dir
    worker_name = driver_args.worker_name

    # If caching is enabled, try to get previous execution results and directly
    # use as output.
    if enable_cache:
      output_result = self._get_output_from_previous_run(
          input_dict, output_dict, exec_properties, driver_args)
      if output_result:
        tf.logging.info('Found cache from previous run.')
        return data_types.ExecutionDecision(
            input_dict=input_dict,
            output_dict=output_result,
            exec_properties=exec_properties)

    # Previous run is not available, prepare execution.
    # Registers execution in metadata.
    execution_id = self._metadata_handler.prepare_execution(
        worker_name, exec_properties)
    tf.logging.info('Preparing new execution.')

    # Checks inputs exist and have valid states and locks them to avoid GC half
    # way
    self._verify_artifacts(input_dict)

    # Updates output.
    for name, output_list in output_dict.items():
      for artifact in output_list:
        artifact.uri = self._generate_output_uri(artifact, base_output_dir,
                                                 name, execution_id)

    return data_types.ExecutionDecision(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        execution_id=execution_id)

  # TODO(ruoyu): Deprecate this in favor of pre_execution() once migration to
  # go/tfx-oss-artifact-passing finishes.
  def prepare_execution(
      self,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      output_dict: Dict[Text, List[types.TfxArtifact]],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
  ) -> data_types.ExecutionDecision:
    """Prepares inputs, outputs and execution properties for actual execution.

    This method could be overridden by custom drivers if they have a different
    logic. The default behavior is to check ml.metadata for an existing
    execution of same inputs and exec_properties, and use previous outputs
    instead of a new execution if found.

    Args:
      input_dict: key -> TfxArtifact for inputs. One can expect every input
        already registered in ML metadata except ExamplesGen.
      output_dict: key -> TfxArtifact for outputs. Uris of the outputs are not
        assigned. It's subclasses' responsibility to set the real output uris.
      exec_properties: Dict of other execution properties.
      driver_args: An instance of DriverArgs class.

    Returns:
      data_types.ExecutionDecision object.

    Raises:
      RuntimeError: if any input as an empty uri.
    """
    tf.logging.info('Enter driver.')
    self._log_properties(input_dict, output_dict, exec_properties)
    execution_decision = self._default_caching_handling(input_dict, output_dict,
                                                        exec_properties,
                                                        driver_args)
    tf.logging.info('Prepared execution.')
    self._log_properties(execution_decision.input_dict,
                         execution_decision.output_dict,
                         execution_decision.exec_properties)
    return execution_decision

  def resolve_input_artifacts(
      self,
      input_dict: Dict[Text, channel.Channel],
      pipeline_info: data_types.PipelineInfo,
  ) -> Dict[Text, List[types.TfxArtifact]]:
    """Resolve input artifacts from metadata.

    Subclasses might override this function for customized artifact properties
    resoultion logic.

    Args:
      input_dict: key -> Channel mapping for inputs generated in logical
        pipeline.
      pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
        related properties including component_type and component_id.

    Returns:
      Final execution properties that will be used in execution.

    Raises:
      RuntimeError: for Channels that do not contain any artifact. This will be
      reverted once we support Channel-based input resolution.
    """
    result = {}
    for name, input_channel in input_dict.items():
      artifacts = list(input_channel.get())
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
      component_info: data_types.ComponentInfo,  # pylint: disable=unused-argument
  ) -> Dict[Text, Any]:
    """Resolve execution properties.

    Subclasses might override this function for customized execution properties
    resoultion logic.

    Args:
      exec_properties: Original execution properties passed in.
      component_info: An instance of data_types.ComponentInfo, holding component
        related properties including component_type and component_id.

    Returns:
      Final execution properties that will be used in execution.
    """
    return exec_properties

  def _prepare_output_artifacts(
      self,
      output_dict: Dict[Text, channel.Channel],
      execution_id: int,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> Dict[Text, List[types.TfxArtifact]]:
    """Prepare output artifacts by assigning uris to each artifact."""
    result = channel.unwrap_channel_dict(output_dict)
    base_output_dir = os.path.join(pipeline_info.pipeline_root,
                                   component_info.component_id)
    for name, output_list in result.items():
      for artifact in output_list:
        artifact.uri = self._generate_output_uri(artifact, base_output_dir,
                                                 name, execution_id)
    return result

  def _fetch_cached_artifacts(self, output_dict: Dict[Text, channel.Channel],
                              cached_execution_id: int
                             ) -> Dict[Text, List[types.TfxArtifact]]:
    """Fetch cached output artifacts."""
    output_artifacts_dict = channel.unwrap_channel_dict(output_dict)
    return self._metadata_handler.fetch_previous_result_artifacts(
        output_artifacts_dict, cached_execution_id)

  def pre_execution(
      self,
      input_dict: Dict[Text, channel.Channel],
      output_dict: Dict[Text, channel.Channel],
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
    input_artifacts = self.resolve_input_artifacts(input_dict, pipeline_info)
    tf.logging.info('Resolved input artifacts are: {}'.format(input_artifacts))
    # Step 2. Register execution in metadata.
    execution_id = self._metadata_handler.register_execution(
        exec_properties=exec_properties,
        pipeline_info=pipeline_info,
        component_info=component_info)
    tf.logging.info('Execution id of the upcoming component execution is %s',
                    execution_id)
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
        tf.logging.info('Found cached_execution: %s', cached_execution_id)
        # Step 4b. New execution not needed. Fetch cached output artifacts.
        try:
          output_artifacts = self._fetch_cached_artifacts(
              output_dict=output_dict, cached_execution_id=cached_execution_id)
          tf.logging.info('Cached output artifacts are: %s', output_artifacts)
          use_cached_results = True
        except RuntimeError:
          tf.logging.warning('Error when trying to get cached output artifacts')
          use_cached_results = False
    if not use_cached_results:
      tf.logging.info('Cached results not found, move on to new execution')
      # Step 4a. New execution is needed. Prepare output artifacts.
      output_artifacts = self._prepare_output_artifacts(
          output_dict=output_dict,
          execution_id=execution_id,
          pipeline_info=pipeline_info,
          component_info=component_info)
      tf.logging.info(
          'Output artifacts skeleton for the upcoming execution are: %s',
          output_artifacts)
      exec_properties = self.resolve_exec_properties(exec_properties,
                                                     component_info)
      tf.logging.info('Execution properties for the upcoming execution are: %s',
                      exec_properties)

    return data_types.ExecutionDecision(input_artifacts, output_artifacts,
                                        exec_properties, execution_id,
                                        use_cached_results)
