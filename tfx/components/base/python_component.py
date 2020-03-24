# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Native Python function component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
from typing import Any, Dict, List, Text

# Standard Imports

import absl
import six

from tfx import types as tfx_types
from tfx.components.base.base_component import _SimpleComponent
from tfx.components.base.base_executor import BaseExecutor
from tfx.components.base.executor_spec import ExecutorClassSpec
from tfx.components.base.function_parser import ArgFormats
from tfx.components.base.function_parser import parse_typehint_component_function
from tfx.types.component_spec import ChannelParameter


class _FunctionExecutor(BaseExecutor):
  """Base class for function-based executors."""

  # Properties that should be overridden by subclass.
  _ARG_FORMATS = {}
  _FUNCTION = lambda: None
  _RETURNED_VALUES = {}

  def Do(self, input_dict: Dict[Text, List[tfx_types.Artifact]],
         output_dict: Dict[Text, List[tfx_types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    function_args = []
    for arg_format, name in self._ARG_FORMATS:
      if arg_format == ArgFormats.INPUT_ARTIFACT:
        function_args.append(input_dict[name][0])
      elif arg_format == ArgFormats.INPUT_ARTIFACT_URI:
        function_args.append(input_dict[name][0].uri)
      elif arg_format == ArgFormats.OUTPUT_ARTIFACT:
        function_args.append(output_dict[name][0])
      elif arg_format == ArgFormats.OUTPUT_ARTIFACT_URI:
        function_args.append(output_dict[name][0].uri)
      elif arg_format == ArgFormats.ARTIFACT_VALUE:
        function_args.append(input_dict[name][0].value)
      else:
        raise ValueError('Unknown argument format: %r' % (arg_format,))

    # Call function and check returned values.
    outputs = self._FUNCTION(*function_args)
    outputs = outputs or {}
    if not isinstance(outputs, dict):
      absl.logging.warning(
          'Expected component executor function to return a dict of outputs.')
      return

    # Assign returned ValueArtifact values.
    for name in self._RETURNED_VALUES:
      if name not in outputs:
        absl.logging.warning(
            'Did not receive expected output %r as return value from '
            'component.', name)
        continue
      output_dict[name][0].value = outputs[name]


def component_from_typehints(func: types.FunctionType):
  """Decorator creating a component from a typehint-annotated Python function.

  Experimental: no backwards compatibility guarantees.

  Args:
    func: Typehint-annotated component executor function.

  Returns:
    BaseComponent subclass for the guven component executor function.

  Raises:
    EnvironmentError: if the current Python interpreter is not Python 3.
  """
  if six.PY2:
    raise EnvironmentError(
        'component_from_typehints() is only supported in Python 3.')

  inputs, outputs, arg_formats, returned_values = (
      parse_typehint_component_function(func))

  channel_inputs = {}
  channel_outputs = {}
  for key, artifact_type in inputs.items():
    channel_inputs[key] = ChannelParameter(type=artifact_type)
  for key, artifact_type in outputs.items():
    channel_outputs[key] = ChannelParameter(type=artifact_type)
  component_spec = type(
      '%s_Spec' % func.__name__,
      (tfx_types.ComponentSpec,),
      {
          'INPUTS': channel_inputs,
          'OUTPUTS': channel_outputs,
          # TODO(ccy): add support for execution properties or remove
          # execution properties from the SDK, merging them with component
          # inputs.
          'PARAMETERS': {},
      })

  executor_class = type(
      '%s_Executor' % func.__name__, (_FunctionExecutor,), {
          '_ARG_FORMATS': arg_formats,
          '_FUNCTION': func,
          '_RETURNED_VALUES': returned_values,
      })

  # TODO(ccy): Extend ExecutorClassSpec to serialize function for remote
  # execution.
  executor_spec = ExecutorClassSpec(executor_class=executor_class)

  return type(
      '%s_Component' % func.__name__, (_SimpleComponent,), {
          'SPEC_CLASS': component_spec,
          'EXECUTOR_SPEC': executor_spec,
          '_INPUTS': inputs,
          '_OUTPUTS': outputs,
      })
