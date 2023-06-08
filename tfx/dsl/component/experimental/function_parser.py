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

import inspect
import types
from typing import Any, Dict, Optional, Tuple, Type, Union, get_args, get_origin

from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental import json_compat
from tfx.dsl.component.experimental import utils
from tfx.types import artifact
from tfx.types import standard_artifacts

try:
  import apache_beam as beam  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


_PRIMITIVE_TO_ARTIFACT = {
    int: standard_artifacts.Integer,
    float: standard_artifacts.Float,
    str: standard_artifacts.String,
    bytes: standard_artifacts.Bytes,
    bool: standard_artifacts.Boolean,
}


# Map from `Optional[T]` to `T` for primitive types. This map is a simple way
# to extract the value of `T` from its optional typehint, since the internal
# fields of the typehint vary depending on the Python version.
_OPTIONAL_PRIMITIVE_MAP = dict((Optional[t], t) for t in _PRIMITIVE_TO_ARTIFACT)


def _validate_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[str, Any],
    subject_message: str,
) -> None:
  """Validates signature of a typehint-annotated component executor function."""
  utils.assert_no_varargs_varkw(argspec)

  # Validate argument type hints.
  for arg in argspec.args:
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
    typehints: Dict[str, Any],
) -> Tuple[
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, utils.ArgFormats],
    Dict[str, bool],
    Dict[str, Any],
    Dict[str, Any],
]:
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
      (one of `int`, `float`, `str`, `bool`, `beam.Pipeline` or any json
      compatible types). Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `utils.ArgFormats` enum).
    arg_defaults: Dictionary mapping names of optional arguments to default
      values.
    returned_outputs: A dictionary mapping output names that are declared as
      ValueArtifact returned outputs to whether the output was declared
      Optional (and thus has a nullable return value).
    json_typehints: A dictionary mapping input names that is declared as
      a json compatible type to the annotation.
    return_json_typehints: A dictionary mapping output names that is declared as
      a json compatible type to the annotation.
  """
  # Extract optional arguments as dict from name to its declared optional value.
  arg_defaults = utils.extract_arg_defaults(argspec)

  # Parse function arguments.
  inputs = {}
  outputs = {}
  parameters = {}
  arg_formats = {}
  returned_outputs = {}
  json_typehints = {}
  return_json_typehints = {}
  for arg in argspec.args:
    arg_typehint = typehints[arg]
    # If the typehint is `Optional[T]` for a primitive type `T`, unwrap it.
    if arg_typehint in _OPTIONAL_PRIMITIVE_MAP:
      arg_typehint = _OPTIONAL_PRIMITIVE_MAP[arg_typehint]
    if isinstance(arg_typehint, annotations.InputArtifact):
      if arg_defaults.get(arg, None) is not None:
        raise ValueError(
            'If an input artifact is declared as an optional argument, '
            'its default value must be `None` (got default value %r for '
            'input argument %r of %r instead).' % (arg_defaults[arg], arg, func)
        )
      artifact_type = arg_typehint.type
      # If the typehint is InputArtifact[List[Artifact]]], unwrap it.
      if artifact_type is list or get_origin(artifact_type) is list:
        artifact_type = get_args(artifact_type)[0]
        arg_formats[arg] = utils.ArgFormats.LIST_INPUT_ARTIFACTS
      else:
        arg_formats[arg] = utils.ArgFormats.INPUT_ARTIFACT
      inputs[arg] = artifact_type
    elif isinstance(arg_typehint, annotations.OutputArtifact):
      if arg in arg_defaults:
        raise ValueError(
            'Output artifact of component function cannot be declared as '
            'optional (error for argument %r of %r).' % (arg, func)
        )
      arg_formats[arg] = utils.ArgFormats.OUTPUT_ARTIFACT
      outputs[arg] = arg_typehint.type
    elif isinstance(arg_typehint, annotations.Parameter):
      utils.parse_parameter_arg(
          arg,
          arg_defaults,
          arg_typehint,
          parameters,
          arg_formats,
      )
    elif isinstance(arg_typehint, annotations.BeamComponentParameter):
      if arg in arg_defaults and arg_defaults[arg] is not None:
        raise ValueError(
            'The default value for BeamComponentParameter must be None.'
        )
      arg_formats[arg] = utils.ArgFormats.BEAM_PARAMETER
      parameters[arg] = arg_typehint.type
    elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
      if arg in arg_defaults:
        if not (
            arg_defaults[arg] is None
            or isinstance(arg_defaults[arg], arg_typehint)
        ):
          raise ValueError(
              'The default value for optional input value %r on function %r '
              'must be an instance of its declared type %r or `None` (got %r '
              'instead)' % (arg, func, arg_typehint, arg_defaults[arg])
          )
      arg_formats[arg] = utils.ArgFormats.ARTIFACT_VALUE
      inputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
    elif json_compat.is_json_compatible(arg_typehint):
      json_typehints[arg] = arg_typehint
      arg_formats[arg] = utils.ArgFormats.ARTIFACT_VALUE
      inputs[arg] = standard_artifacts.JsonValue
    elif inspect.isclass(arg_typehint) and issubclass(
        arg_typehint, artifact.Artifact
    ):
      raise ValueError(
          'Invalid type hint annotation for argument %r on function %r. '
          'Argument with an artifact class typehint annotation should indicate '
          'whether it is used as an input or output artifact by using the '
          '`InputArtifact[ArtifactType]` or `OutputArtifact[ArtifactType]` '
          'typehint annotations.' % (arg, func)
      )
    else:
      raise ValueError(
          'Unknown type hint annotation for argument %r on function %r'
          % (arg, func)
      )

  if 'return' in typehints and typehints['return'] not in (None, type(None)):
    for arg, arg_typehint in typehints['return'].kwargs.items():
      if arg_typehint in _OPTIONAL_PRIMITIVE_MAP:
        unwrapped_typehint = _OPTIONAL_PRIMITIVE_MAP[arg_typehint]
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[unwrapped_typehint]
        returned_outputs[arg] = True
      elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
        returned_outputs[arg] = False
      elif json_compat.is_json_compatible(arg_typehint):
        outputs[arg] = standard_artifacts.JsonValue
        return_json_typehints[arg] = arg_typehint
        # check if Optional
        returned_outputs[arg] = get_origin(arg_typehint) is Union and type(
            None
        ) in get_args(arg_typehint)
      else:
        raise ValueError(
            'Unknown type hint annotation %r for returned output %r on '
            'function %r' % (arg_typehint, arg, func)
        )

  return (
      inputs,
      outputs,
      parameters,
      arg_formats,
      arg_defaults,
      returned_outputs,
      json_typehints,
      return_json_typehints,
  )


def parse_typehint_component_function(
    func: types.FunctionType,
) -> Tuple[
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Any],
    Dict[str, utils.ArgFormats],
    Dict[str, Any],
    Dict[str, bool],
    Dict[str, Any],
    Dict[str, Any],
]:
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
      (one of `int`, `float`, `str`, `bool`, `beam.Pipeline` or any json
      compatible types). Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `utils.ArgFormats` enum).
    arg_defaults: Dictionary mapping names of optional arguments to default
      values.
    returned_outputs: A dictionary mapping output names that are declared as
      ValueArtifact returned outputs to whether the output was declared
      Optional (and thus has a nullable return value).
    json_typehints: A dictionary mapping input names that are declared as json
      compatible types to the annotation.
    return_json_typehints: A dictionary mapping output names that are declared
      as json compatible types to the annotation.
  """
  utils.assert_is_functype(func)

  # Inspect the component executor function.
  typehints = inspect.get_annotations(func)
  argspec = inspect.getfullargspec(func)  # pytype: disable=module-attr
  subject_message = 'Component declared as a typehint-annotated function'
  _validate_signature(func, argspec, typehints, subject_message)

  # Parse the function and return its details.
  (inputs, outputs, parameters, arg_formats, arg_defaults, returned_outputs,
   json_typehints, return_json_typehints) = (
       _parse_signature(func, argspec, typehints))

  return (inputs, outputs, parameters, arg_formats, arg_defaults,
          returned_outputs, json_typehints, return_json_typehints)
