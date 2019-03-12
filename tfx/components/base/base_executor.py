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
from future.utils import with_metaclass
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text
from tfx.utils import types


class BaseExecutor(with_metaclass(abc.ABCMeta, object)):
  """Abstract TFX executor class."""

  @abc.abstractmethod
  def Do(self, input_dict,
         output_dict,
         exec_properties):
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

  def __init__(self, beam_pipeline_args = None):
    """Constructs a beam based executor.

    Args:
      beam_pipeline_args: Beam pipeline args.
    """
    self._beam_pipeline_args = beam_pipeline_args

  # TODO(b/126182711): Look into how to support fusion of multiple executors
  # into same pipeline.
  def _get_beam_pipeline_args(self):
    """Get beam pipeline args."""
    return self._beam_pipeline_args

  def _log_startup(self, inputs,
                   outputs,
                   exec_properties):
    """Log inputs, outputs, and executor properties in a standard format."""
    tf.logging.info('Starting {} execution.'.format(self.__class__.__name__))
    tf.logging.info('Inputs for {} is: {}'.format(
        self.__class__.__name__, types.jsonify_tfx_type_dict(inputs)))
    tf.logging.info('Outputs for {} is: {}'.format(
        self.__class__.__name__, types.jsonify_tfx_type_dict(outputs)))
    tf.logging.info('Execution properties for {} is: {}'.format(
        self.__class__.__name__, json.dumps(exec_properties)))
