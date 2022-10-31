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

import abc
import json
import os
from typing import Any, Dict, List, Optional

from absl import logging
from tfx import types
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils

try:
  import apache_beam as beam  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


class BaseExecutor(abc.ABC):
  """Abstract TFX executor class."""

  class Context:
    """A context class for all excecutors."""

    def __init__(self,
                 extra_flags: Optional[List[str]] = None,
                 tmp_dir: Optional[str] = None,
                 unique_id: Optional[str] = None,
                 executor_output_uri: Optional[str] = None,
                 stateful_working_dir: Optional[str] = None,
                 pipeline_node: Optional[pipeline_pb2.PipelineNode] = None,
                 pipeline_info: Optional[pipeline_pb2.PipelineInfo] = None,
                 pipeline_run_id: Optional[str] = None):
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
      # The config of this Node.
      self._pipeline_node = pipeline_node
      # The config of the pipeline that this node is running in.
      self._pipeline_info = pipeline_info
      # The id of the pipeline run that this execution is in.
      self._pipeline_run_id = pipeline_run_id

    def get_tmp_path(self) -> str:
      if not self._tmp_dir or not self._unique_id:
        raise RuntimeError('Temp path not available')
      return os.path.join(self._tmp_dir, str(self._unique_id), '')

    @property
    def executor_output_uri(self) -> str:
      return self._executor_output_uri

    @property
    def stateful_working_dir(self) -> str:
      return self._stateful_working_dir

    @property
    def pipeline_node(self) -> pipeline_pb2.PipelineNode:
      return self._pipeline_node

    @property
    def pipeline_info(self) -> pipeline_pb2.PipelineInfo:
      return self._pipeline_info

    @property
    def pipeline_run_id(self) -> str:
      return self._pipeline_run_id

  @abc.abstractmethod
  def Do(  # pylint: disable=invalid-name
      self, input_dict: Dict[str, List[types.Artifact]],
      output_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, Any],
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

  def _get_tmp_dir(self) -> str:
    """Get the temporary directory path."""
    if not self._context:
      raise RuntimeError('No context for the executor')
    tmp_path = self._context.get_tmp_path()
    if not fileio.exists(tmp_path):
      logging.info('Creating temp directory at %s', tmp_path)
      fileio.makedirs(tmp_path)
    return tmp_path

  def _log_startup(self, inputs: Dict[str, List[types.Artifact]],
                   outputs: Dict[str, List[types.Artifact]],
                   exec_properties: Dict[str, Any]) -> None:
    """Log inputs, outputs, and executor properties in a standard format."""
    logging.debug('Starting %s execution.', self.__class__.__name__)
    logging.debug('Inputs for %s are: %s', self.__class__.__name__,
                  artifact_utils.jsonify_artifact_dict(inputs))
    logging.debug('Outputs for %s are: %s', self.__class__.__name__,
                  artifact_utils.jsonify_artifact_dict(outputs))
    logging.debug(
        'Execution properties for %s are: %s', self.__class__.__name__,
        json.dumps(
            data_types_utils.build_value_dict(
                data_types_utils.build_metadata_value_dict(exec_properties))))


class EmptyExecutor(BaseExecutor):
  """An empty executor that does nothing."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    pass
