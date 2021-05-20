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

# TODO(ccy): Remove pytype "disable=attribute-error" and "disable=module-attr"
# overrides after Python 2 support is removed from TFX.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import inspect
import types
from typing import Any, Dict, Optional, Set, Text, Tuple, Type, Union

from tfx.dsl.component.experimental import annotations
from tfx.types import artifact
from tfx.types import standard_artifacts


class ArgFormats(enum.Enum):
  INPUT_ARTIFACT = 1
  OUTPUT_ARTIFACT = 2
  ARTIFACT_VALUE = 3
  PARAMETER = 4


_PRIMITIVE_TO_ARTIFACT = {
    int: standard_artifacts.Integer,
    float: standard_artifacts.Float,
    Text: standard_artifacts.String,
    bytes: standard_artifacts.Bytes,
}


# Map from `Optional[T]` to `T` for primitive types. This map is a simple way
# to extract the value of `T` from its optional typehint, since the internal
# fields of the typehint vary depending on the Python version.
_OPTIONAL_PRIMITIVE_MAP = dict((Optional[t], t) for t in _PRIMITIVE_TO_ARTIFACT)


def _validate_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[Text, Any],
    subject_message: Text) -> None:
  """Validates signature of a typehint-annotated component executor function."""
  args, varargs, keywords = argspec.args, argspec.varargs, argspec.varkw
  if varargs or keywords:
    raise ValueError('%s does not support *args or **kwargs arguments.' %
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
          (inspect.isclass(arg_typehint) and
           issubclass(arg_typehint, artifact.Artifact))):
        raise ValueError(
            ('Output artifacts for the component executor function %r should '
             'be declared as function parameters annotated with type hint '
             '`tfx.types.annotations.OutputArtifact[T]` where T is a '
             'subclass of `tfx.types.Artifact`. They should not be declared '
             'as part of the return value `OutputDict` type hint.') % func)
  elif 'return' not in typehints or typehints['return'] in (None, type(None)):
    pass
  else:
    raise ValueError(
        ('%s must have either an OutputDict instance or `None` as its return '
         'value typehint.') % subject_message)


def _parse_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[Text, Any]
) -> Tuple[Dict[Text, Type[artifact.Artifact]], Dict[
    Text, Type[artifact.Artifact]], Dict[Text, Type[Union[
        int, float, Text, bytes]]], Dict[Text, Any], Dict[Text, ArgFormats],
           Set[Text]]:
  """Parses signature of a typehint-annotated component executor function.

  Args:
    func: A component executor function to be parsed.
    argspec: A `inspect.FullArgSpec` instance describing the component executor
      function. Usually obtained from `inspect.getfullargspec(func)`.
    typehints: A dictionary mapping function argument names to type hints.
      Usually obtained from `func.__annotations__`.

  Returns:
    inputs: A dictionary mapping each input name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    outputs: A dictionary mapping each output name to its artifact type (as a
      subclass of `tfx.types.Artifact`).
    parameters: A dictionary mapping each parameter name to its primitive type
      (one of `int`, `float`, `Text` and `bytes`).
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `ArgFormats` enum).
    arg_defaults: Dictionary mapping names of optional arguments to default
      values.
    returned_outputs: A set of output names that are declared as ValueArtifact
      returned outputs.
  """
  # Extract optional arguments as dict from name to its declared optional value.
  arg_defaults = {}
  if argspec.defaults:
    arg_defaults = dict(
        zip(argspec.args[-len(argspec.defaults):], argspec.defaults))

  # Parse function arguments.
  inputs = {}
  outputs = {}
  parameters = {}
  arg_formats = {}
  returned_outputs = set()
  for arg in argspec.args:
    arg_typehint = typehints[arg]
    # If the typehint is `Optional[T]` for a primitive type `T`, unwrap it.
    if arg_typehint in _OPTIONAL_PRIMITIVE_MAP:
      arg_typehint = _OPTIONAL_PRIMITIVE_MAP[arg_typehint]
    if isinstance(arg_typehint, annotations.InputArtifact):
      if arg_defaults.get(arg, None) is not None:
        raise ValueError(
            ('If an input artifact is declared as an optional argument, '
             'its default value must be `None` (got default value %r for '
             'input argument %r of %r instead).') %
            (arg_defaults[arg], arg, func))
      arg_formats[arg] = ArgFormats.INPUT_ARTIFACT
      inputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.OutputArtifact):
      if arg in arg_defaults:
        raise ValueError(
            ('Output artifact of component function cannot be declared as '
             'optional (error for argument %r of %r).') % (arg, func))
      arg_formats[arg] = ArgFormats.OUTPUT_ARTIFACT
      outputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.Parameter):
      if arg in arg_defaults:
        if not (arg_defaults[arg] is None or
                isinstance(arg_defaults[arg], arg_typehint.type)):
          raise ValueError((
              'The default value for optional parameter %r on function %r must '
              'be an instance of its declared type %r or `None` (got %r '
              'instead)') % (arg, func, arg_typehint.type, arg_defaults[arg]))
      arg_formats[arg] = ArgFormats.PARAMETER
      parameters[arg] = arg_typehint.type
    elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
      if arg in arg_defaults:
        if not (arg_defaults[arg] is None or
                isinstance(arg_defaults[arg], arg_typehint)):
          raise ValueError(
              ('The default value for optional input value %r on function %r '
               'must be an instance of its declared type %r or `None` (got %r '
               'instead)') % (arg, func, arg_typehint, arg_defaults[arg]))
      arg_formats[arg] = ArgFormats.ARTIFACT_VALUE
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

  if 'return' in typehints and typehints['return'] not in (None, type(None)):
    for arg, arg_typehint in typehints['return'].kwargs.items():
      if arg_typehint in _PRIMITIVE_TO_ARTIFACT:
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
        returned_outputs.add(arg)
      else:
        raise ValueError(
            ('Unknown type hint annotation %r for returned output %r on '
             'function %r') % (arg_typehint, arg, func))

  return (inputs, outputs, parameters, arg_formats, arg_defaults,
          returned_outputs)


def parse_typehint_component_function(
    func: types.FunctionType
) -> Tuple[Dict[Text, Type[artifact.Artifact]], Dict[
    Text, Type[artifact.Artifact]], Dict[Text, Type[Union[
        int, float, Text, bytes]]], Dict[Text, Any], Dict[Text, ArgFormats],
           Set[Text]]:
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
    parameters: A dictionary mapping each parameter name to its primitive type
      (one of `int`, `float`, `Text` and `bytes`).
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `ArgFormats` enum).
    arg_defaults: Dictionary mapping names of optional arguments to default
      values.
    returned_outputs: A set of output names that are declared as ValueArtifact
      returned outputs.
  """
  # Check input argument type.
  if not isinstance(func, types.FunctionType):
    raise ValueError(
        'Expected a typehint-annotated Python function (got %r instead).' %
        (func,))

  # Inspect the component executor function.
  typehints = func.__annotations__  # pytype: disable=attribute-error
  argspec = inspect.getfullargspec(func)  # pytype: disable=module-attr
  subject_message = 'Component declared as a typehint-annotated function'
  _validate_signature(func, argspec, typehints, subject_message)

  # Parse the function and return its details.
  inputs, outputs, parameters, arg_formats, arg_defaults, returned_outputs = (
      _parse_signature(func, argspec, typehints))

  return (inputs, outputs, parameters, arg_formats, arg_defaults,
          returned_outputs)
