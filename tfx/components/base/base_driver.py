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

  def _verify_inputs(self,
                     input_dict: Dict[Text, List[types.TfxArtifact]]) -> None:
    """Verify input exist.

    Args:
      input_dict: key -> TfxArtifact for inputs.

    Raises:
      RuntimeError: if any input as an empty uri.
    """
    for single_input_list in input_dict.values():
      for single_input in single_input_list:
        if not single_input.uri:
          raise RuntimeError('Input {} not available'.format(single_input))
        if not tf.gfile.Exists(os.path.dirname(single_input.uri)):
          raise RuntimeError('Input {} is missing'.format(single_input))

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
    self._verify_inputs(input_dict)

    # Updates output.
    max_input_span = 0
    for input_list in input_dict.values():
      for single_input in input_list:
        max_input_span = max(max_input_span, single_input.span)
    # TODO(ruoyu): This location is dangerous because this function is not
    # guaranteed to be called on custom driver.
    for output_name, output_list in output_dict.items():
      for output_artifact in output_list:
        # Updates outputs uri based on execution id and optional split.
        # Last empty string forces this be to a directory.
        output_artifact.uri = os.path.join(base_output_dir, output_name,
                                           str(execution_id),
                                           output_artifact.split, '')
        if tf.gfile.Exists(output_artifact.uri):
          msg = 'Output artifact uri {} already exists'.format(
              output_artifact.uri)
          tf.logging.error(msg)
          raise RuntimeError(msg)
        else:
          # TODO(zhitaoli): Consider refactoring this out into something
          # which can handle permission bits.
          tf.logging.info('Creating output artifact uri %s as directory',
                          output_artifact.uri)
          tf.gfile.MakeDirs(output_artifact.uri)
        # Defaults to make the output span the max of input span.
        output_artifact.span = max_input_span

    return data_types.ExecutionDecision(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        execution_id=execution_id)

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
