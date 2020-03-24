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
"""Component executor function parser."""

# TODO(ccy): Remove pytype overrides after Python 2 support is removed from TFX.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Dict, List, Set, Text, Tuple, Type

# Standard Imports

import six

from tfx import types as tfx_types
from tfx.components.base.base_component import _SimpleComponent
from tfx.components.base.executor_spec import ExecutorContainerSpec
from tfx.components.base.function_parser import ArgFormats
from tfx.components.base.function_parser import parse_typehint_component_function
from tfx.components.base.serialization import SourceCopySerializer
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter


def _get_stub_template():
  contents = open(
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          'container_entrypoint_stub.py')).read()
  return re.search('# BEGIN_STUB\n([^\0]+\n)# END_STUB', contents).group(1)  # pytype: disable=attribute-error


def _construct_entrypoint_command(function_name: Text,
                                  function_definition_code: Text,
                                  arg_formats: List[Tuple[ArgFormats, Text]],
                                  outputs: Dict[Text, Type[tfx_types.Artifact]],
                                  returned_values: Set[Text]):
  """Construct container entrypoint `python -c ...` command and arguments."""
  parameter_args = []
  parameter_list_code_parts = []
  for arg_format, name in arg_formats:
    parameter_list_code_parts.append('(%r, ArgFormats.%s)' %
                                     (name, arg_format.name))
    if arg_format == ArgFormats.INPUT_ARTIFACT:
      parameter_args.append('{{input_dict[%r][0]}}' % name)
    elif arg_format == ArgFormats.INPUT_ARTIFACT_URI:
      parameter_args.append('{{input_dict[%r][0].uri}}' % name)
    elif arg_format == ArgFormats.OUTPUT_ARTIFACT:
      parameter_args.append('{{output_dict[%r][0]}}' % name)
    elif arg_format == ArgFormats.OUTPUT_ARTIFACT_URI:
      parameter_args.append('{{output_dict[%r][0].uri}}' % name)
    elif arg_format == ArgFormats.ARTIFACT_VALUE:
      parameter_args.append('{{input_dict[%r][0].value}}' % name)
    else:
      raise ValueError('Unknown argument format: %r' % (arg_format,))

  return_values_code_parts = []
  for name in returned_values:
    output_type = outputs[name]
    if output_type == standard_artifacts.IntegerType:
      return_values_code_parts.append('%r: int' % name)
    elif output_type == standard_artifacts.FloatType:
      return_values_code_parts.append('%r: float' % name)
    elif output_type == standard_artifacts.StringType:
      return_values_code_parts.append('%r: Text' % name)
    elif output_type == standard_artifacts.BytesType:
      return_values_code_parts.append('%r: bytes' % name)
    else:
      raise ValueError(
          ('The custom ValueArtifact type %s is not supported as a return type '
           'for @container_component_from_typehints.') % output_type)

  parameter_list_code = '[%s]' % ', '.join(parameter_list_code_parts)
  return_values_code = '{%s}' % ', '.join(return_values_code_parts)

  code_to_insert = '\n'.join([
      '%s\n' % function_definition_code,
      '_COMPONENT_NAME = %r' % function_name,
      '_COMPONENT_FUNCTION = %s' % function_name,
      '_COMPONENT_ARGS = %s' % parameter_list_code,
      '_COMPONENT_RETURN_VALUE_TYPES = %s' % return_values_code,
  ])
  code_blob = _get_stub_template().replace('# INSERTION_POINT', code_to_insert)

  command = 'python'
  args = ['-c', code_blob] + parameter_args

  return command, args


def container_component_from_typehints(image: Text):
  """Decorator creating a container-based component from Python function.

  Experimental: no backwards compatibility guarantees.

  Args:
    image: The Docker container image to use for this component. This image must
      contain Python 3 (and right now, Tensorflow, for file I/O).

  Returns:
    Decorator that should be applied to a component executor function.

  Raises:
    EnvironmentError: if the current Python interpreter is not Python 3.
  """
  if six.PY2:
    raise EnvironmentError(
        'container_component_from_typehints() is only supported in Python 3.')

  def _inner(func):
    """Inner decorator function for container_component_from_typehints."""
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

    function_name = func.__name__
    function_definition = SourceCopySerializer.encode(func)
    command, args = _construct_entrypoint_command(function_name,
                                                  function_definition,
                                                  arg_formats, outputs,
                                                  returned_values)
    executor_spec = ExecutorContainerSpec(
        image=image, command=[command], args=args)

    return type('%s_Component' % func.__name__, (_SimpleComponent,), {
        'SPEC_CLASS': component_spec,
        'EXECUTOR_SPEC': executor_spec,
    })

  return _inner
