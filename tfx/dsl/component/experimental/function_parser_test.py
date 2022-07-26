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
"""Tests for tfx.dsl.components.base.function_parser."""

from typing import Dict, List, Optional, Union

import apache_beam as beam
import tensorflow as tf
from tfx.dsl.component.experimental.annotations import BeamComponentParameter
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.function_parser import ArgFormats
from tfx.dsl.component.experimental.function_parser import check_strict_json_compat
from tfx.dsl.component.experimental.function_parser import is_json_compatible
from tfx.dsl.component.experimental.function_parser import parse_typehint_component_function
from tfx.types import standard_artifacts

_X = Union['_Jsonable', int, float, str, bool, type(None)]
_Jsonable = Union[List['_X'], Dict[str, '_X'], '_Jsonable']
_InvalidA = Dict[str, '_InvalidA']
_InvalidB = List['_InvalidB']
_InvalidC = Union['_InvalidC', int]
_InvalidD = Union[Dict[str, '_InvalidD'], int, str]

_ValidA = Optional[List['_ValidA']]  # Can be a list of None.
_ValidB = Union['_ValidB', Dict[str, float]]

_TypeA = Dict[str, Union['_TypeA', int]]
_TypeB = Dict[str, Union['_TypeB', int]]
_TypeC = Dict[str, '_TypeC']
_TypeD = Optional[List[Union['_TypeD', int]]]  # Can be a list of None.
_TypeE = Union['_TypeE', Dict[str, float]]
_TypeF = Optional[List[Union['_TypeF']]]


class FunctionParserTest(tf.test.TestCase):

  def testSimpleFunctionParse(self):

    def func_a(
        a: int,
        b: int,
        unused_c: str,
        unused_d: bytes,
        unused_e: _Jsonable,
        unused_f: Parameter[float],
        unused_g: BeamComponentParameter[beam.Pipeline] = None
    ) -> OutputDict(
        c=float, d=_Jsonable):
      return {'c': float(a + b), 'd': unused_e}

    (inputs, outputs, parameters, arg_formats, arg_defaults, returned_values,
     json_typehints, return_json_typehints) = (
         parse_typehint_component_function(func_a))
    self.assertDictEqual(
        inputs, {
            'a': standard_artifacts.Integer,
            'b': standard_artifacts.Integer,
            'unused_c': standard_artifacts.String,
            'unused_d': standard_artifacts.Bytes,
            'unused_e': standard_artifacts.JsonValue,
        })
    self.assertDictEqual(outputs, {
        'c': standard_artifacts.Float,
        'd': standard_artifacts.JsonValue,
    })
    self.assertDictEqual(parameters, {
        'unused_f': float,
        'unused_g': beam.Pipeline,
    })
    self.assertDictEqual(
        arg_formats, {
            'a': ArgFormats.ARTIFACT_VALUE,
            'b': ArgFormats.ARTIFACT_VALUE,
            'unused_c': ArgFormats.ARTIFACT_VALUE,
            'unused_d': ArgFormats.ARTIFACT_VALUE,
            'unused_e': ArgFormats.ARTIFACT_VALUE,
            'unused_f': ArgFormats.PARAMETER,
            'unused_g': ArgFormats.BEAM_PARAMETER,
        })
    self.assertDictEqual(arg_defaults, {'unused_g': None})
    self.assertEqual(returned_values, {'c': False, 'd': False})
    self.assertDictEqual(json_typehints, {'unused_e': _Jsonable})
    self.assertDictEqual(return_json_typehints, {'d': _Jsonable})

  def testArtifactFunctionParse(self):

    def func_a(
        examples: InputArtifact[standard_artifacts.Examples],
        model: OutputArtifact[standard_artifacts.Model],
        schema: InputArtifact[standard_artifacts.Schema],
        statistics: OutputArtifact[standard_artifacts.ExampleStatistics],
        num_steps: Parameter[int],
        unused_beam_pipeline: BeamComponentParameter[beam.Pipeline],
    ) -> OutputDict(
        precision=float,
        recall=float,
        message=str,
        serialized_value=bytes,
        is_blessed=bool):
      del examples, model, schema, statistics, num_steps, unused_beam_pipeline
      return {
          'precision': 0.9,
          'recall': 0.8,
          'message': 'foo',
          'serialized_value': b'bar',
          'is_blessed': False,
      }

    (inputs, outputs, parameters, arg_formats, arg_defaults, returned_values,
     json_typehints, return_json_typehints) = (
         parse_typehint_component_function(func_a))
    self.assertDictEqual(
        inputs, {
            'examples': standard_artifacts.Examples,
            'schema': standard_artifacts.Schema,
        })
    self.assertDictEqual(
        outputs, {
            'model': standard_artifacts.Model,
            'statistics': standard_artifacts.ExampleStatistics,
            'precision': standard_artifacts.Float,
            'recall': standard_artifacts.Float,
            'message': standard_artifacts.String,
            'serialized_value': standard_artifacts.Bytes,
            'is_blessed': standard_artifacts.Boolean,
        })
    self.assertDictEqual(parameters, {
        'num_steps': int,
        'unused_beam_pipeline': beam.Pipeline,
    })
    self.assertDictEqual(
        arg_formats, {
            'examples': ArgFormats.INPUT_ARTIFACT,
            'model': ArgFormats.OUTPUT_ARTIFACT,
            'schema': ArgFormats.INPUT_ARTIFACT,
            'statistics': ArgFormats.OUTPUT_ARTIFACT,
            'num_steps': ArgFormats.PARAMETER,
            'unused_beam_pipeline': ArgFormats.BEAM_PARAMETER,
        })
    self.assertDictEqual(arg_defaults, {})
    self.assertEqual(
        returned_values, {
            'precision': False,
            'recall': False,
            'message': False,
            'serialized_value': False,
            'is_blessed': False,
        })
    self.assertDictEqual(json_typehints, {})
    self.assertDictEqual(return_json_typehints, {})

  def testEmptyReturnValue(self):
    # No output typehint.
    def func_a(examples: InputArtifact[standard_artifacts.Examples],
               model: OutputArtifact[standard_artifacts.Model], a: int,
               b: float, c: Parameter[int], d: Parameter[str],
               e: Parameter[bytes], f: BeamComponentParameter[beam.Pipeline]):
      del examples, model, a, b, c, d, e, f

    # `None` output typehint.
    def func_b(examples: InputArtifact[standard_artifacts.Examples],
               model: OutputArtifact[standard_artifacts.Model], a: int,
               b: float, c: Parameter[int], d: Parameter[str],
               e: Parameter[bytes],
               f: BeamComponentParameter[beam.Pipeline]) -> None:
      del examples, model, a, b, c, d, e, f

    # Both functions should be parsed in the same way.
    for func in [func_a, func_b]:
      (inputs, outputs, parameters, arg_formats,
       arg_defaults, returned_values, json_typehints,
       return_json_typehints) = parse_typehint_component_function(func)
      self.assertDictEqual(
          inputs, {
              'examples': standard_artifacts.Examples,
              'a': standard_artifacts.Integer,
              'b': standard_artifacts.Float,
          })
      self.assertDictEqual(outputs, {
          'model': standard_artifacts.Model,
      })
      self.assertDictEqual(parameters, {
          'c': int,
          'd': str,
          'e': bytes,
          'f': beam.Pipeline,
      })
      self.assertDictEqual(
          arg_formats, {
              'examples': ArgFormats.INPUT_ARTIFACT,
              'model': ArgFormats.OUTPUT_ARTIFACT,
              'a': ArgFormats.ARTIFACT_VALUE,
              'b': ArgFormats.ARTIFACT_VALUE,
              'c': ArgFormats.PARAMETER,
              'd': ArgFormats.PARAMETER,
              'e': ArgFormats.PARAMETER,
              'f': ArgFormats.BEAM_PARAMETER,
          })
      self.assertDictEqual(arg_defaults, {})
      self.assertEqual(returned_values, {})
      self.assertDictEqual(json_typehints, {})
      self.assertDictEqual(return_json_typehints, {})

  def testOptionalArguments(self):
    # Various optional argument schemes.
    def func_a(a: float,
               b: int,
               c: Parameter[str],
               d: int = 123,
               e: Optional[int] = 345,
               f: str = 'abc',
               g: bytes = b'xyz',
               h: bool = False,
               i: Parameter[str] = 'default',
               j: Parameter[int] = 999,
               k: BeamComponentParameter[beam.Pipeline] = None,
               examples: InputArtifact[standard_artifacts.Examples] = None,
               optional_json: Optional[Union[List[Dict[str, int]],
                                             Dict[str, bool]]] = None):
      del a, b, c, d, e, f, g, h, i, j, k, examples, optional_json

    (inputs, outputs, parameters, arg_formats, arg_defaults, returned_values,
     json_typehints, return_json_typehints) = (
         parse_typehint_component_function(func_a))
    self.assertDictEqual(
        inputs,
        {
            'a': standard_artifacts.Float,
            'b': standard_artifacts.Integer,
            # 'c' is missing here as it is a parameter.
            'd': standard_artifacts.Integer,
            'e': standard_artifacts.Integer,
            'f': standard_artifacts.String,
            'g': standard_artifacts.Bytes,
            'h': standard_artifacts.Boolean,
            # 'i' is missing here as it is a parameter.
            # 'j' is missing here as it is a parameter.
            'examples': standard_artifacts.Examples,
            'optional_json': standard_artifacts.JsonValue,
        })
    self.assertDictEqual(outputs, {})
    self.assertDictEqual(parameters, {
        'c': str,
        'i': str,
        'j': int,
        'k': beam.Pipeline,
    })
    self.assertDictEqual(
        arg_formats, {
            'a': ArgFormats.ARTIFACT_VALUE,
            'b': ArgFormats.ARTIFACT_VALUE,
            'c': ArgFormats.PARAMETER,
            'd': ArgFormats.ARTIFACT_VALUE,
            'e': ArgFormats.ARTIFACT_VALUE,
            'f': ArgFormats.ARTIFACT_VALUE,
            'g': ArgFormats.ARTIFACT_VALUE,
            'h': ArgFormats.ARTIFACT_VALUE,
            'i': ArgFormats.PARAMETER,
            'j': ArgFormats.PARAMETER,
            'k': ArgFormats.BEAM_PARAMETER,
            'examples': ArgFormats.INPUT_ARTIFACT,
            'optional_json': ArgFormats.ARTIFACT_VALUE,
        })
    self.assertDictEqual(
        arg_defaults, {
            'd': 123,
            'e': 345,
            'f': 'abc',
            'g': b'xyz',
            'h': False,
            'i': 'default',
            'j': 999,
            'k': None,
            'examples': None,
            'optional_json': None,
        })
    self.assertEqual(returned_values, {})
    self.assertDictEqual(json_typehints, {
        'optional_json': Optional[Union[List[Dict[str, int]], Dict[str, bool]]]
    })
    self.assertDictEqual(return_json_typehints, {})

  def testOptionalReturnValues(self):

    def func_a() -> OutputDict(
        precision=float,
        recall=float,
        message=str,
        serialized_value=bytes,
        optional_label=Optional[str],
        optional_metric=Optional[float],
        optional_json=Optional[_Jsonable]):
      return {
          'precision': 0.9,
          'recall': 0.8,
          'message': 'foo',
          'serialized_value': b'bar',
          'optional_label': None,
          'optional_metric': 1.0,
          'optional_json': {
              'foo': 1
          },
      }

    (inputs, outputs, parameters, arg_formats, arg_defaults, returned_values,
     json_typehints, return_json_typehints) = (
         parse_typehint_component_function(func_a))
    self.assertDictEqual(inputs, {})
    self.assertDictEqual(
        outputs, {
            'precision': standard_artifacts.Float,
            'recall': standard_artifacts.Float,
            'message': standard_artifacts.String,
            'serialized_value': standard_artifacts.Bytes,
            'optional_label': standard_artifacts.String,
            'optional_metric': standard_artifacts.Float,
            'optional_json': standard_artifacts.JsonValue,
        })
    self.assertDictEqual(parameters, {})
    self.assertDictEqual(arg_formats, {})
    self.assertDictEqual(arg_defaults, {})
    self.assertEqual(
        returned_values, {
            'precision': False,
            'recall': False,
            'message': False,
            'serialized_value': False,
            'optional_label': True,
            'optional_metric': True,
            'optional_json': True,
        })
    self.assertDictEqual(json_typehints, {})
    self.assertDictEqual(return_json_typehints,
                         {'optional_json': Optional[_Jsonable]})

  def testFunctionParseErrors(self):
    # Non-function arguments.
    with self.assertRaisesRegex(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function(object())
    with self.assertRaisesRegex(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function('foo')

    # Unannotated lambda.
    with self.assertRaisesRegex(
        ValueError, 'must have all arguments annotated with typehints'):
      parse_typehint_component_function(lambda x: True)

    # Function with *args and **kwargs.
    with self.assertRaisesRegex(
        ValueError,
        'must have either an OutputDict instance or `None` as its return'):

      def func_a(a: int, b: int) -> object:
        del a, b
        return object()

      parse_typehint_component_function(func_a)

    # Function with *args and **kwargs.
    with self.assertRaisesRegex(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_b(a: int, b: int, *unused_args) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_b)
    with self.assertRaisesRegex(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_c(a: int, b: int, **unused_kwargs) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_c)

    # Not all arguments annotated with typehints.
    with self.assertRaisesRegex(
        ValueError, 'must have all arguments annotated with typehint'):

      def func_d(a: int, b) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_d)

    # Artifact type used in annotation without `InputArtifact[ArtifactType]` or
    # `OutputArtifact[ArtifactType]` wrapper.
    with self.assertRaisesRegex(
        ValueError, 'Invalid type hint annotation.*'
        'should indicate whether it is used as an input or output artifact'):

      def func_e(a: int,
                 unused_b: standard_artifacts.Examples) -> OutputDict(c=float):
        return {'c': float(a)}

      parse_typehint_component_function(func_e)

    # Invalid input typehint.
    with self.assertRaisesRegex(ValueError, 'Unknown type hint annotation'):

      def func_f(a: int, b: Dict[int, int]) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_f)

    # Invalid output typehint.
    with self.assertRaisesRegex(ValueError, 'Unknown type hint annotation'):

      def func_g(a: int, b: int) -> OutputDict(c='whatever'):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_g)

    # Invalid output typehint.
    with self.assertRaisesRegex(ValueError, 'Unknown type hint annotation'):

      def func_h(a: int,
                 b: int) -> OutputDict(c=Optional[standard_artifacts.Examples]):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_h)

    # Output artifact in the wrong place.
    with self.assertRaisesRegex(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_i(a: int, b: int) -> OutputDict(c=standard_artifacts.Examples):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_i)
    with self.assertRaisesRegex(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_j(
          a: int,
          b: int) -> OutputDict(c=OutputArtifact[standard_artifacts.Examples]):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_j)

    # Input artifact declared optional with non-`None` default value.
    with self.assertRaisesRegex(
        ValueError,
        'If an input artifact is declared as an optional argument, its default '
        'value must be `None`'):

      def func_k(
          a: int,
          b: int,
          examples: InputArtifact[standard_artifacts.Examples] = 123
      ) -> OutputDict(c=float):
        del examples
        return {'c': float(a + b)}

      parse_typehint_component_function(func_k)

    # Output artifact declared optional.
    with self.assertRaisesRegex(
        ValueError,
        'Output artifact of component function cannot be declared as optional'):

      def func_l(
          a: int,
          b: int,
          model: OutputArtifact[standard_artifacts.Model] = None
      ) -> OutputDict(c=float):
        del model
        return {'c': float(a + b)}

      parse_typehint_component_function(func_l)

    # Optional parameter's default value does not match declared type.
    with self.assertRaisesRegex(
        ValueError,
        'The default value for optional input value .* on function .* must be '
        'an instance of its declared type .* or `None`'):

      def func_m(a: int,
                 b: int,
                 num_iterations: int = 'abc') -> OutputDict(c=float):
        del num_iterations
        return {'c': float(a + b)}

      parse_typehint_component_function(func_m)

    # Optional parameter's default value does not match declared type.
    with self.assertRaisesRegex(
        ValueError,
        'The default value for optional parameter .* on function .* must be an '
        'instance of its declared type .* or `None`'):

      def func_n(a: int,
                 b: int,
                 num_iterations: Parameter[int] = 'abc') -> OutputDict(c=float):
        del num_iterations
        return {'c': float(a + b)}

      parse_typehint_component_function(func_n)

  def testIsJsonCompatible(self):
    def func():
      pass
    for typehint in (
        Dict[str, float], List[int], Dict[str, List[Dict[str, int]]],
        Optional[Dict[str, Dict[str, bool]]],
        Optional[Dict[str, Optional[List[int]]]],
        Union[Dict[str, int], type(None)], Union[Dict[str, str], List[bool]],
        _Jsonable, _ValidA, _ValidB):
      self.assertTrue(is_json_compatible(typehint, func))
    for typehint in (
        # Bare primitives.
        dict, Dict, Union,
        # Invalid Dict, Union or List parameters.
        Dict[str, Dict], Dict[str, bytes], Dict[int, float],
        Union[Dict[str, int], float], List[bytes], List['Y'],
        # Primitive types.
        int, str, float, dict, bytes, bool, type(None),
        # Invalid ForwardRef.
        _InvalidA, _InvalidB, _InvalidC, _InvalidD):
      self.assertFalse(is_json_compatible(typehint, func))

  def testCheckStrictJsonCompat(self):
    # Construction time type check.
    for pair in (
        # Pairs are currently valid but not supported by is_json_compatible.
        (int, int), (List, List), (Dict, Dict), (List[str], List),
        (Dict[str, int], Dict), (_TypeC, _TypeB),
        # Valid pairs are supported by is_json_compatible.
        (Dict[str, float], Dict[str, float]), (List[bool], List[bool]),
        (Dict[str, Dict[str, int]], _TypeA), (_TypeA, _TypeB),
        (_TypeB, _Jsonable), (_TypeF, _TypeF), (_TypeF, Optional[_Jsonable]),
        (type(None), _TypeF), (_TypeF, _TypeD), (_TypeE, _Jsonable),
        (_TypeE, Dict[str, float]), (Dict[str, float], _TypeE),
        (_TypeD, _TypeD), (List[int], _TypeD)):
      self.assertTrue(check_strict_json_compat(pair[0], pair[1], __name__))

    for pair in (
        (_TypeA, Dict[str, Dict[str, int]]),
        (Dict[str, Dict[str, int]], _TypeC),
        (_TypeC, Dict[str, Dict[str, int]]), (_TypeB, _TypeC),
        (str, int), (Dict[str, int], Dict[str, str]),
        (Dict, Dict[str, int]), (List, List[float]), (None, int),
        (_TypeD, _TypeF), (_TypeD, List[int]), (_TypeD, _TypeE),
        (_TypeE, _TypeD), (_TypeE, _TypeF), (_TypeF, _TypeE)
    ):
      self.assertFalse(check_strict_json_compat(pair[0], pair[1], __name__))

    # Runtime type check.
    for pair in (({
        'a': [1, 2, 3],
        'b': [3,],
        'c': [1, 2, 3, 4]
    }, Dict[str, List[int]]), ({
        'a': [1, 2, 3.],
        'b': [3,],
        'd': [1, 2, 3, 4]
    }, Dict[str, List[Union[int, float]]]), ({
        'a': {
            'b': True,
            'c': False
        },
        'b': None
    }, Dict[str, Optional[Dict[str, bool]]]),
                 ([1, {
                     'a': True
                 }, None, True,
                   [3., 4.]], List[Optional[Union[int, Dict[str, bool],
                                                  List[float], bool]]]), ({
                                                      'a': [1, 2, 3]
                                                  }, Union[List[int],
                                                           Dict[str, List[int]],
                                                           Dict[str, float]]),
                 ([1, 2, 3], Union[List[int], Dict[str, List[int]],
                                   Dict[str, float]]), ({
                                       'a': 1.
                                   }, Union[List[int], Dict[str, List[int]],
                                            Dict[str,
                                                 float]]), ([1, 2,
                                                             3], _Jsonable),
                 ([], _Jsonable), ({}, _Jsonable), (None, Optional[_Jsonable]),
                 ([None, None], _ValidA), ([None, None], _ValidA), ({
                     'a': 1.,
                     'b': 2.
                 }, _ValidB), ({
                     'a': {
                         'a': 3
                     },
                     'c': {
                         'c': {
                             'c': 1
                         }
                     }
                 }, _TypeA), (None, _TypeD), ([1, 2], _TypeD), ([None], _TypeD),
                 ([[1, 2], [[3]], [4]], _TypeD), ({
                     'a': 1.,
                     'b': 2.
                 }, _TypeE)):
      self.assertTrue(check_strict_json_compat(pair[0], pair[1], __name__))

    for pair in (({
        'a': [1, 2, 3.]
    }, Dict[str, List[int]]), ({
        'a': {
            'b': True,
            'c': False
        },
        'b': 1
    }, Dict[str, Optional[Dict[str, bool]]]), ({
        'a': [True, False]
    }, Union[List[int], Dict[str, List[int]],
             Dict[str, float]]), ([b'123'], _Jsonable), ({
                 1: 2
             }, _Jsonable), ([{
                 'a': b'b'
             }], _Jsonable), (None, _Jsonable), ([1, 2], _ValidA), ({
                 'a': True,
                 'b': False
             }, _ValidB), ({
                 'a': {
                     'a': 3
                 },
                 'c': [1, 2]
             }, _TypeA), ({
                 'a': 1
             }, _TypeC), ([1.], _TypeD), ({
                 'a': True,
                 'b': 2.
             }, _TypeE)):
      self.assertFalse(check_strict_json_compat(pair[0], pair[1], __name__))

if __name__ == '__main__':
  tf.test.main()
