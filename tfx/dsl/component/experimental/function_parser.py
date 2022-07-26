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

import enum
import inspect
import itertools
import sys
import types
from typing import Any, Dict, Optional, Tuple, Type, Union

from tfx.dsl.component.experimental import annotations
from tfx.types import artifact
from tfx.types import standard_artifacts

try:
  import apache_beam as beam  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


class ArgFormats(enum.Enum):
  INPUT_ARTIFACT = 1
  OUTPUT_ARTIFACT = 2
  ARTIFACT_VALUE = 3
  PARAMETER = 4
  BEAM_PARAMETER = 5


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
_JSONABLE_PRIMITIVES = frozenset([int, float, str, bool, type(None)])


def is_json_compatible(
    typehint: Type,  # pylint: disable=g-bare-generic
    func: Optional[types.FunctionType] = None
) -> bool:
  """Check if a type hint represents a JSON-compatible type.

  Currently, 'JSON-compatible' means a type conforms with T, where T is defined
  as `X = Union['T', int, float, str, bool, NoneType]`,
  `T = Union[List['X'], Dict[str, 'X'], T]`, It can be Optional.

  Args:
    typehint: The typehint to check. It can accept `ForwardRef` types. Note that
      all `ForwardRef` types must be defined at the top level of the same Python
      module as where func is defined.
    func: The function annotated by typehint.
  Returns:
    True if typehint is a JSON-compatible type.
  """
  has_leaf = False   # Used to handle invalid types like T = Dict[str, 'T'].
  seen = set()
  def check(typehint: Any, not_primitive: bool = True) -> bool:
    # Return False for [int, str, float, bytes, bool, type(None)]
    # if not_primitive is True.
    nonlocal has_leaf
    if typehint.__class__.__name__ == 'ForwardRef':
      typename = typehint.__forward_arg__
      if typename in seen:
        return True
      module_name = getattr(func, '__module__', None)
      globaln = getattr(sys.modules.get(module_name, None), '__dict__', {})
      seen.add(typename)
      return check(
          typehint=globaln.get(typename, None), not_primitive=not_primitive)
    origin = getattr(typehint, '__origin__', typehint)
    args = getattr(typehint, '__args__', None)
    if origin is dict or origin is list or origin is Union:
      # Starting from Python 3.9 Dict won't have default args (~KT, ~VT)
      # and List won't have default args (~T).
      if not args:
        return False
      elif origin is dict and args[0] is not str:
        return False
      elif origin is dict and args[0] is str:
        return check(typehint=args[1], not_primitive=False)
      # Handle top level optional.
      elif origin is Union and not_primitive:
        return all([
            arg is type(None) or
            check(typehint=arg, not_primitive=True) for arg in args
        ])
      else:
        return all([check(typehint=arg, not_primitive=False) for arg in args])
    else:
      res = origin in _JSONABLE_PRIMITIVES
      if res and not not_primitive:
        has_leaf = True
      return not not_primitive and res
  return check(typehint) and has_leaf


def check_strict_json_compat(
    in_type: Type, expect_type: Type, module: str) -> bool:  # pylint: disable=g-bare-generic
  """Check if in_type conforms with expect_type.

  Args:
    in_type: Input type hint. It can be any JSON-serializable `Dict[str, T]`
      `List[T]` or `ForwardRef`.
    expect_type: Expected type hint. It can be any JSON-serializable
      `Dict[str, X]`, `List` or `ForwardRef`.
    module: Module name used to look up `ForwardRef`.

  Returns:
    True if in_type is valid w.r.t. expect_type.
  """
  seen_forwardref = {}
  check_instance = False
  if getattr(in_type, '__module__', None) not in {'typing', 'builtins'}:
    check_instance = True

  # Skip circular states when doing all and any.
  def _all(iter_args, seen_pairs):
    for arg in iter_args:
      if check_instance:
        pair = (id(arg[0]), arg[1])
      else:
        pair = arg
      if pair in seen_pairs:
        continue
      elif not _helper(*arg, seen_pairs):
        return False
    return True

  def _any(iter_args, seen_pairs):
    for arg in iter_args:
      if check_instance:
        pair = (id(arg[0]), arg[1])
      else:
        pair = arg
      if pair in seen_pairs:
        continue
      elif _helper(*arg, seen_pairs):
        return True
    return False

  def _resolve_forwardref(typename):
    """Turning ForwardRef typename into the reference type."""
    resolved_typehint = None
    if typename in seen_forwardref:
      resolved_typehint = seen_forwardref[typename]
    else:
      globaln = getattr(sys.modules.get(module, None), '__dict__', {})
      resolved_typehint = globaln.get(typename, None)
      if resolved_typehint:
        seen_forwardref[typename] = resolved_typehint
    return resolved_typehint

  def _helper(in_type, expect_type, seen_pairs):
    new_seen_pairs = seen_pairs.copy()
    in_obj = None
    if check_instance:
      in_obj, in_type = in_type, type(in_type)

    resolve_in, resolve_expect = None, None
    if in_type.__class__.__name__ == 'ForwardRef':
      resolve_in = _resolve_forwardref(in_type.__forward_arg__)
      if not resolve_in:
        return False
    if expect_type.__class__.__name__ == 'ForwardRef':
      resolve_expect = _resolve_forwardref(expect_type.__forward_arg__)
      if not resolve_expect:
        return False

    if resolve_in or resolve_expect:
      if check_instance:
        new_seen_pairs.add((id(in_obj), expect_type))
      else:
        new_seen_pairs.add((in_type, expect_type))
      in_type = resolve_in if resolve_in else in_type
      expect_type = resolve_expect if resolve_expect else expect_type

    in_args = getattr(in_type, '__args__', ())
    in_origin = getattr(in_type, '__origin__', in_type)
    expect_args = getattr(expect_type, '__args__', ())
    expect_origin = getattr(expect_type, '__origin__', expect_type)

    if in_origin is Union:
      return _all(zip(in_args, itertools.repeat(expect_type)), new_seen_pairs)
    if expect_origin is Union:
      if check_instance:
        return _any(zip(itertools.repeat(in_obj), expect_args), new_seen_pairs)
      else:
        return _any(zip(itertools.repeat(in_type), expect_args), new_seen_pairs)
    if in_origin != expect_origin:
      return False
    elif in_origin in (
        dict, list
    ) and expect_args and expect_args[0].__class__.__name__ == 'TypeVar':
      return True
    elif check_instance:
      if isinstance(in_obj, list):
        return not expect_args or _all(
            zip(in_obj, itertools.repeat(expect_args[0])),
            new_seen_pairs)
      elif isinstance(in_obj, dict):
        return not expect_args or (_all(
            zip(in_obj.keys(), itertools.repeat(expect_args[0])),
            new_seen_pairs) and _all(
                zip(in_obj.values(), itertools.repeat(expect_args[1])),
                new_seen_pairs))
      else:
        return True
    # For List -> List[X] and Dict -> Dict[X, Y].
    elif len(in_args) < len(expect_args):
      return False
    # For Python 3.7, where Dict and List have args KT, KV, T. Return True
    # whenever the expect type is Dict or List.
    else:
      return _all(zip(in_args, expect_args), new_seen_pairs)
  return _helper(in_type, expect_type, set())


def _validate_signature(
    func: types.FunctionType,
    argspec: inspect.FullArgSpec,  # pytype: disable=module-attr
    typehints: Dict[str, Any],
    subject_message: str) -> None:
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
    typehints: Dict[str, Any]
) -> Tuple[
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[Union[int, float, str, bytes, _BeamPipeline]]],
    Dict[str, Any],
    Dict[str, ArgFormats],
    Dict[str, bool],
    Dict[str, Any],
    Dict[str, Any]]:
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
      (one of `int`, `float`, `Text`, `bytes` and `beam.Pipeline`).
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `ArgFormats` enum).
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
  arg_defaults = {}
  if argspec.defaults:
    arg_defaults = dict(
        zip(argspec.args[-len(argspec.defaults):], argspec.defaults))

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
    elif isinstance(arg_typehint, annotations.BeamComponentParameter):
      if arg in arg_defaults and arg_defaults[arg] is not None:
        raise ValueError('The default value for BeamComponentParameter must '
                         'be None.')
      arg_formats[arg] = ArgFormats.BEAM_PARAMETER
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
    elif is_json_compatible(arg_typehint, func):
      json_typehints[arg] = arg_typehint
      arg_formats[arg] = ArgFormats.ARTIFACT_VALUE
      inputs[arg] = standard_artifacts.JsonValue
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
      if arg_typehint in _OPTIONAL_PRIMITIVE_MAP:
        unwrapped_typehint = _OPTIONAL_PRIMITIVE_MAP[arg_typehint]
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[unwrapped_typehint]
        returned_outputs[arg] = True
      elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
        outputs[arg] = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
        returned_outputs[arg] = False
      elif is_json_compatible(arg_typehint, func):
        outputs[arg] = standard_artifacts.JsonValue
        return_json_typehints[arg] = arg_typehint
        # check if Optional
        origin = getattr(arg_typehint, '__origin__', None)
        args = getattr(arg_typehint, '__args__', None)
        returned_outputs[arg] = origin is Union and type(None) in args
      else:
        raise ValueError(
            ('Unknown type hint annotation %r for returned output %r on '
             'function %r') % (arg_typehint, arg, func))

  return (inputs, outputs, parameters, arg_formats, arg_defaults,
          returned_outputs, json_typehints, return_json_typehints)


def parse_typehint_component_function(
    func: types.FunctionType
) -> Tuple[
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[artifact.Artifact]],
    Dict[str, Type[Union[int, float, str, bytes]]],
    Dict[str, Any],
    Dict[str, ArgFormats],
    Dict[str, bool],
    Dict[str, Any],
    Dict[str, Any]]:
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
      (one of `int`, `float`, `Text`, `bytes` and `beam.Pipeline`).
    arg_formats: Dictionary representing the input arguments of the given
      component executor function. Each entry's key is the argument's string
      name; each entry's value is the format of the argument to be passed into
      the function (given by a value of the `ArgFormats` enum).
    arg_defaults: Dictionary mapping names of optional arguments to default
      values.
    returned_outputs: A dictionary mapping output names that are declared as
      ValueArtifact returned outputs to whether the output was declared
      Optional (and thus has a nullable return value).
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
  (inputs, outputs, parameters, arg_formats, arg_defaults, returned_outputs,
   json_typehints, return_json_typehints) = (
       _parse_signature(func, argspec, typehints))

  return (inputs, outputs, parameters, arg_formats, arg_defaults,
          returned_outputs, json_typehints, return_json_typehints)
