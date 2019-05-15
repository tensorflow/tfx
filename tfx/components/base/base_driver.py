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

import logging
import os
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text

from tfx.orchestration import metadata
from tfx.utils import types


class ExecutionDecision(object):
  """ExecutionDecision records how executor should perform next execution.

  Attributes:
    input_dict: Updated key -> TfxType for inputs that will be used by
      actual execution.
    output_dict: Updated key -> TfxType for outputs that will be used by
      actual execution.
    exec_properties: Updated dict of other execution properties that will be
      used by actual execution.
    execution_id: Registered execution_id for the upcoming execution. If
      None, then no execution needed.
  """

  def __init__(self,
               input_dict,
               output_dict,
               exec_properties,
               execution_id = None):
    self.input_dict = input_dict
    self.output_dict = output_dict
    self.exec_properties = exec_properties
    self.execution_id = execution_id


class DriverOptions(object):
  """Options to driver from orchestration system.

  Args:
    worker_name: orchestrator specific instance name for the worker running
      current component.
    base_output_dir: common base directory shared by all components in current
      pipeline execution.
    enable_cache: whether cache is enabled in current execution.
  """

  def __init__(self, worker_name, base_output_dir,
               enable_cache):
    self.worker_name = worker_name
    self.base_output_dir = base_output_dir
    self.enable_cache = enable_cache


class BaseDriver(object):
  """BaseDriver is the base class of all custom drivers.

  This can also be used as the default driver of a component if no custom logic
  is needed.

  Args:
    logger: A logging.Logger
    metadata_handler: An instance of Metadata.
  """

  def __init__(self, logger,
               metadata_handler):
    self._metadata_handler = metadata_handler
    self._logger = logger

  def _log_properties(self, input_dict,
                      output_dict,
                      exec_properties):
    """Log inputs, outputs, and executor properties in a standard format."""
    self._logger.info('Starting %s driver.', self.__class__.__name__)
    self._logger.info('Inputs for {} is: {}'.format(self.__class__.__name__,
                                                    input_dict))
    self._logger.info('Execution properties for {} is: {}'.format(
        self.__class__.__name__, exec_properties))
    self._logger.info('Outputs for {} is: {}'.format(self.__class__.__name__,
                                                     output_dict))

  def _get_output_from_previous_run(
      self,
      input_dict,
      output_dict,
      exec_properties,
      driver_options,
  ):
    """Returns outputs from previous identical execution if found."""
    previous_execution_id = self._metadata_handler.previous_run(
        type_name=driver_options.worker_name,
        input_dict=input_dict,
        exec_properties=exec_properties)
    if previous_execution_id:
      final_output = self._metadata_handler.fetch_previous_result_artifacts(
          output_dict, previous_execution_id)
      for output_list in final_output.values():
        for single_output in output_list:
          if not single_output.uri or not tf.gfile.Exists(single_output.uri):
            self._logger.warn(
                'URI of cached artifact %s does not exist, forcing new execution',
                single_output)
            return None
      self._logger.info(
          'Reusing previous execution {} output artifacts {}'.format(
              previous_execution_id, final_output))
      return final_output
    else:
      return None

  def _verify_inputs(
      self, input_dict):
    """Verify input exist.

    Args:
      input_dict: key -> TfxType for inputs.

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
      input_dict,
      output_dict,
      exec_properties,
      driver_options,
  ):
    """Check cache for desired and applicable identical execution."""
    enable_cache = driver_options.enable_cache
    base_output_dir = driver_options.base_output_dir
    worker_name = driver_options.worker_name

    # If caching is enabled, try to get previous execution results and directly
    # use as output.
    if enable_cache:
      output_result = self._get_output_from_previous_run(
          input_dict, output_dict, exec_properties, driver_options)
      if output_result:
        self._logger.info('Found cache from previous run.')
        return ExecutionDecision(
            input_dict=input_dict,
            output_dict=output_result,
            exec_properties=exec_properties)

    # Previous run is not available, prepare execution.
    # Registers execution in metadata.
    execution_id = self._metadata_handler.prepare_execution(
        worker_name, exec_properties)
    self._logger.info('Preparing new execution.')

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
          tf.logging.warn('Output artifact uri %s already exists',
                          output_artifact.uri)
        else:
          # TODO(zhitaoli): Consider refactoring this out into something
          # which can handle permission bits.
          tf.logging.info('Creating output artifact uri %s as directory',
                          output_artifact.uri)
          tf.gfile.MakeDirs(output_artifact.uri)
        # Defaults to make the output span the max of input span.
        output_artifact.span = max_input_span

    return ExecutionDecision(
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        execution_id=execution_id)

  def prepare_execution(
      self,
      input_dict,
      output_dict,
      exec_properties,
      driver_options,
  ):
    """Prepares inputs, outputs and execution properties for actual execution.

    This method could be overridden by custom drivers if they have a different
    logic. The default behavior is to check ml.metadata for an existing
    execution of same inputs and exec_properties, and use previous outputs
    instead of a new execution if found.

    Args:
      input_dict: key -> TfxType for inputs. One can expect every input already
        registered in ML metadata except ExamplesGen.
      output_dict: key -> TfxType for outputs. Uris of the outputs are not
        assigned. It's subclasses' responsibility to set the real output uris.
      exec_properties: Dict of other execution properties.
      driver_options: An instance of DriverOptions class.

    Returns:
      ExecutionDecision object.

    Raises:
      RuntimeError: if any input as an empty uri.
    """
    self._logger.info('Enter driver.')
    self._log_properties(input_dict, output_dict, exec_properties)
    execution_decision = self._default_caching_handling(
        input_dict, output_dict, exec_properties, driver_options)
    self._logger.info('Prepared execution.')
    self._log_properties(execution_decision.input_dict,
                         execution_decision.output_dict,
                         execution_decision.exec_properties)
    return execution_decision
