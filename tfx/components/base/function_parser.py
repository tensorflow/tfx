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
"""Component executor function parser.

Internal use only. No backwards compatibility guarantees.
"""

# TODO(ccy): Remove "pytype: disable=module-attr" overrides after Python 2
# support is removed from TFX.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import inspect
import types
import typing
from typing import Any, Dict, List, Set, Text, Tuple, Type

from tfx.components.base import annotations
from tfx.types import artifact
from tfx.types import standard_artifacts


class ArgFormats(enum.Enum):
  INPUT_ARTIFACT = 1
  INPUT_URI = 2
  OUTPUT_ARTIFACT = 3
  OUTPUT_URI = 4
  ARTIFACT_VALUE = 5


_PRIMITIVE_TO_ARTIFACT = {
    int: standard_artifacts.Integer,
    float: standard_artifacts.Float,
    Text: standard_artifacts.String,
    bytes: standard_artifacts.Bytes,
}


def _validate_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[Text, Any],
    subject_message: Text) -> None:
  """Validates signature of a typehint-annotated component executor function."""
  args, varargs, keywords, defaults = (argspec.args, argspec.varargs,
                                       argspec.varkw, argspec.defaults)
  if varargs or keywords:
    raise ValueError('%s does not support *args or **kwargs arguments.' %
                     subject_message)
  if defaults:
    # TODO(ccy): add support for optional arguments / default values.
    raise ValueError(
        '%s currently does not support optional arguments / default values.' %
        subject_message)

  # Validate argument type hints.
  for arg in args:
    if isinstance(arg, list):
      # Note: this feature was removed in Python 3:
      # https://www.python.org/dev/peps/pep-3113/.
      raise ValueError('%s does not support nested input arguments.' %
                       subject_message)
    if arg not in typehints:
      raise ValueError('%s must have all arguments annotated with typehints.' %
                       subject_message)

  # Validate return type hints.
  if isinstance(typehints.get('return', None), annotations.OutputDict):
    for arg, arg_typehint in typehints['return'].kwargs.items():
      if (isinstance(arg_typehint, annotations.OutputArtifact) or
          isinstance(arg_typehint, annotations.OutputUri) or
          (inspect.isclass(arg_typehint) and
           issubclass(arg_typehint, artifact.Artifact))):
        raise ValueError(
            ('Output artifacts for the component executor function %r should '
             'be declared as function parameters annotated with type hint '
             '`tfx.types.annotations.OutputArtifact[T]` '
             'or `tfx.types.annotations.OutputUri[T]` where T is a '
             'subclass of `tfx.types.Artifact`. They should not be declared '
             'as part of the return value `OutputDict` type hint.') % func)
  elif 'return' not in typehints or typehints['return'] == type(None):
    pass
  else:
    raise ValueError(
        ('%s must have either an OutputDict instance or None as its return '
         'value typehint.') % subject_message)


def _parse_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[Text, Any]
) -> Tuple[Dict[Text, Type[artifact.Artifact]], Dict[
    Text, Type[artifact.Artifact]], List[Tuple[Text, ArgFormats]], Set[Text]]:
  """Parses signature of a typehint-annotated component executor function.

  Args:
    func: A component executor function to be parsed.
    argspec: A `inspect.FullArgSpec` instance describing the component executor
      function. Usually obtained from `inspect.getfullargspec(func)`.
    typehints: A dictionary mapping function argument names to type hints.
      Usually obtained from `typing.get_type_hints(func)`.

  Returns:
    inputs: A dictionary mapping each input name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    outputs: A dictionary mapping each output name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    arg_formats: A list of 2-tuples corresponding (in order) to the input
      arguments of the given component executor function. Each tuple's first
      element is the format of the argument to be passed into the function
      (given by a value of the `ArgFormats` enum); the second element is the
      argument's string name.
    returned_outputs: A set of output names that are declared as ValueArtifact
      returned outputs.
  """
  # Parse function arguments.
  inputs = {}
  outputs = {}
  arg_formats = []
  returned_outputs = set()
  for arg in argspec.args:
    arg_typehint = typehints[arg]
    if isinstance(arg_typehint, annotations.InputArtifact):
      arg_formats.append((arg, ArgFormats.INPUT_ARTIFACT))
      inputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.InputUri):
      arg_formats.append((arg, ArgFormats.INPUT_URI))
      inputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.OutputArtifact):
      arg_formats.append((arg, ArgFormats.OUTPUT_ARTIFACT))
      outputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.OutputUri):
      arg_formats.append((arg, ArgFormats.OUTPUT_URI))
      outputs[arg] = arg_typehint.type
    elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
      arg_formats.append((arg, ArgFormats.ARTIFACT_VALUE))
      inputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
    elif (inspect.isclass(arg_typehint) and
          issubclass(arg_typehint, artifact.Artifact)):
      raise ValueError((
          'Invalid type hint annotation for argument %r on function %r. '
          'Argument with an artifact class typehint annotation should indicate '
          'whether it is used as an input or output artifact by using the '
          '`InputArtifact[ArtifactType]` or `OutputArtifact[ArtifactType]` '
          'typehint annotations.') % (arg, func))
    else:
      raise ValueError(
          'Unknown type hint annotation for argument %r on function %r' %
          (arg, func))

  if 'return' in typehints and typehints['return'] != type(None):
    for arg, arg_typehint in typehints['return'].kwargs.items():
      if arg_typehint in _PRIMITIVE_TO_ARTIFACT:
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
        returned_outputs.add(arg)
      else:
        raise ValueError(
            ('Unknown type hint annotation for returned output %r on function '
             '%r') % (arg, func))

  return inputs, outputs, arg_formats, returned_outputs


def parse_typehint_component_function(
    func: types.FunctionType
) -> Tuple[Dict[Text, Type[artifact.Artifact]], Dict[
    Text, Type[artifact.Artifact]], List[Tuple[Text, ArgFormats]], Set[Text]]:
  """Parses the given component executor function.

  This method parses a typehinted-annotated Python function that is intended to
  be used as a component and returns the information needed about the interface
  (inputs / outputs / returned output values) about that components, as well as
  a list of argument names and formats for determining the parameters that
  should be passed when calling `func(*args)`.

  Args:
    func: A component executor function to be parsed.

  Returns:
    inputs: A dictionary mapping each input name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    outputs: A dictionary mapping each output name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    arg_formats: A list of 2-tuples corresponding (in order) to the input
      arguments of the given component executor function. Each tuple's first
      element is the argument's string name; the second element is the format of
      the argument to be passed into the function (given by a value of the
      `ArgFormats` enum).
    returned_outputs: A set of output names that are declared as ValueArtifact
      returned outputs.
  """
  # Check input argument type.
  if not isinstance(func, types.FunctionType):
    raise ValueError(
        'Expected a typehint-annotated Python function (got %r instead).' %
        (func,))

  # Inspect the component executor function.
  typehints = typing.get_type_hints(func)
  argspec = inspect.getfullargspec(func)  # pytype: disable=module-attr
  subject_message = 'Component declared as a typehint-annotated function'
  _validate_signature(func, argspec, typehints, subject_message)

  # Parse the function and return its details.
  inputs, outputs, arg_formats, returned_outputs = (
      _parse_signature(func, argspec, typehints))
  return inputs, outputs, arg_formats, returned_outputs
