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
from future.utils import with_metaclass
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text
from tfx.utils import deps_utils
from tfx.utils import types


class BaseExecutor(with_metaclass(abc.ABCMeta, object)):
  """Abstract TFX executor class."""

  class Context(object):
    """A context class for all excecutors."""

    def __init__(self, beam_pipeline_args: Optional[List[Text]] = None,
                 tmp_dir: Optional[Text] = None,
                 unique_id: Optional[Text] = None):
      self.beam_pipeline_args = beam_pipeline_args
      # Base temp directory for the pipeline
      self._tmp_dir = tmp_dir
      # A unique id to distinguish every execution run
      self._unique_id = unique_id

    def get_tmp_path(self) -> Text:
      if not self._tmp_dir or not self._unique_id:
        raise RuntimeError('Temp path not available')
      return os.path.join(self._tmp_dir, str(self._unique_id), '')

  @abc.abstractmethod
  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]) -> None:
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
      None.
    """
    pass

  def __init__(self, context: Optional[Context] = None):
    """Constructs a beam based executor."""
    self._context = context
    self._beam_pipeline_args = context.beam_pipeline_args if context else None

    if self._beam_pipeline_args:
      self._beam_pipeline_args = deps_utils.make_beam_dependency_flags(
          self._beam_pipeline_args)

  # TODO(b/126182711): Look into how to support fusion of multiple executors
  # into same pipeline.
  def _get_beam_pipeline_args(self) -> Optional[List[Text]]:
    """Get beam pipeline args."""
    return self._beam_pipeline_args

  def _get_tmp_dir(self) -> Text:
    """Get the temporary directory path."""
    if not self._context:
      raise RuntimeError('No context for the executor')
    tmp_path = self._context.get_tmp_path()
    if not tf.gfile.Exists(tmp_path):
      tf.logging.info('Creating temp directory at %s', tmp_path)
      tf.gfile.MakeDirs(tmp_path)
    return tmp_path

  def _log_startup(self, inputs: Dict[Text, List[types.TfxArtifact]],
                   outputs: Dict[Text, List[types.TfxArtifact]],
                   exec_properties: Dict[Text, Any]) -> None:
    """Log inputs, outputs, and executor properties in a standard format."""
    tf.logging.info('Starting {} execution.'.format(self.__class__.__name__))
    tf.logging.info('Inputs for {} is: {}'.format(
        self.__class__.__name__, types.jsonify_tfx_type_dict(inputs)))
    tf.logging.info('Outputs for {} is: {}'.format(
        self.__class__.__name__, types.jsonify_tfx_type_dict(outputs)))
    tf.logging.info('Execution properties for {} is: {}'.format(
        self.__class__.__name__, json.dumps(exec_properties)))
