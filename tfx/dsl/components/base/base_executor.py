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
"""Abstract TFX executor class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import os
from typing import Any, Dict, List, Optional, Text

from absl import logging
from tfx import types
from tfx.dsl.io import fileio
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact_utils

try:
  import apache_beam as beam  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


class BaseExecutor(abc.ABC):
  """Abstract TFX executor class."""

  class Context(object):
    """A context class for all excecutors."""

    def __init__(self,
                 extra_flags: Optional[List[Text]] = None,
                 tmp_dir: Optional[Text] = None,
                 unique_id: Optional[Text] = None,
                 executor_output_uri: Optional[Text] = None,
                 stateful_working_dir: Optional[Text] = None):
      self.extra_flags = extra_flags
      # Base temp directory for the pipeline
      self._tmp_dir = tmp_dir
      # A unique id to distinguish every execution run
      self._unique_id = unique_id
      # A path for executor to write its output to.
      self._executor_output_uri = executor_output_uri
      # A path to store information for stateful run, e.g. checkpoints for
      # tensorflow trainers.
      self._stateful_working_dir = stateful_working_dir

    def get_tmp_path(self) -> Text:
      if not self._tmp_dir or not self._unique_id:
        raise RuntimeError('Temp path not available')
      return os.path.join(self._tmp_dir, str(self._unique_id), '')

    @property
    def executor_output_uri(self) -> Text:
      return self._executor_output_uri

    @property
    def stateful_working_dir(self) -> Text:
      return self._stateful_working_dir

  @abc.abstractmethod
  def Do(  # pylint: disable=invalid-name
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any],
  ) -> Optional[execution_result_pb2.ExecutorOutput]:
    """Execute underlying component implementation.

    Args:
      input_dict: Input dict from input key to a list of Artifacts. These are
        often outputs of another component in the pipeline and passed to the
        component by the orchestration system.
      output_dict: Output dict from output key to a list of Artifacts. These are
        often consumed by a dependent component.
      exec_properties: A dict of execution properties. These are inputs to
        pipeline with primitive types (int, string, float) and fully
        materialized when a pipeline is constructed. No dependency to other
        component or later injection from orchestration systems is necessary or
        possible on these values.

    Returns:
      execution_result_pb2.ExecutorOutput or None.
    """
    pass

  def __init__(self, context: Optional[Context] = None):
    """Constructs a base executor."""
    self._context = context
    self._extra_flags = context.extra_flags if context else None

  def _get_tmp_dir(self) -> Text:
    """Get the temporary directory path."""
    if not self._context:
      raise RuntimeError('No context for the executor')
    tmp_path = self._context.get_tmp_path()
    if not fileio.exists(tmp_path):
      logging.info('Creating temp directory at %s', tmp_path)
      fileio.makedirs(tmp_path)
    return tmp_path

  def _log_startup(self, inputs: Dict[Text, List[types.Artifact]],
                   outputs: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> None:
    """Log inputs, outputs, and executor properties in a standard format."""
    logging.debug('Starting %s execution.', self.__class__.__name__)
    logging.debug('Inputs for %s are: %s', self.__class__.__name__,
                  artifact_utils.jsonify_artifact_dict(inputs))
    logging.debug('Outputs for %s are: %s', self.__class__.__name__,
                  artifact_utils.jsonify_artifact_dict(outputs))
    logging.debug('Execution properties for %s are: %s',
                  self.__class__.__name__, json.dumps(exec_properties))


class EmptyExecutor(BaseExecutor):
  """An empty executor that does nothing."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    pass
