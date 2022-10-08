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
from tfx.dsl.component.experimental.function_parser import parse_typehint_component_function
from tfx.types import standard_artifacts


class FunctionParserTest(tf.test.TestCase):

  def testSimpleFunctionParse(self):

    def func_a(
        a: int,
        b: int,
        unused_c: str,
        unused_d: bytes,
        unused_e: Dict[str, List[float]],
        unused_f: Parameter[float],
        unused_g: BeamComponentParameter[beam.Pipeline] = None
    ) -> OutputDict(
        c=float, d=Dict[str, List[float]]):
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
    self.assertDictEqual(json_typehints, {'unused_e': Dict[str, List[float]]})
    self.assertDictEqual(return_json_typehints, {'d': Dict[str, List[float]]})

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
        optional_json=Optional[Dict[str, List[bool]]]):
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
                         {'optional_json': Optional[Dict[str, List[bool]]]})

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


if __name__ == '__main__':
  tf.test.main()
