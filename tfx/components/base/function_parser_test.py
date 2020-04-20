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
"""Tests for tfx.components.base.function_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Text
import unittest

# Standard Imports

import six
import tensorflow as tf

from tfx.components.base.annotations import InputArtifact
from tfx.components.base.annotations import InputUri
from tfx.components.base.annotations import OutputArtifact
from tfx.components.base.annotations import OutputDict
from tfx.components.base.annotations import OutputUri
from tfx.components.base.function_parser import ArgFormats
from tfx.components.base.function_parser import parse_typehint_component_function
from tfx.types import standard_artifacts


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class FunctionParserTest(tf.test.TestCase):

  def testSimpleFunctionParse(self):

    def func_a(a: int, b: int, unused_c: Text,
               unused_d: bytes) -> OutputDict(c=float):
      return {'c': float(a + b)}

    inputs, outputs, arg_formats, returned_values = (
        parse_typehint_component_function(func_a))
    self.assertEqual(
        inputs, {
            'a': standard_artifacts.Integer,
            'b': standard_artifacts.Integer,
            'unused_c': standard_artifacts.String,
            'unused_d': standard_artifacts.Bytes,
        })
    self.assertEqual(outputs, {
        'c': standard_artifacts.Float,
    })
    self.assertEqual(arg_formats, [
        ('a', ArgFormats.ARTIFACT_VALUE),
        ('b', ArgFormats.ARTIFACT_VALUE),
        ('unused_c', ArgFormats.ARTIFACT_VALUE),
        ('unused_d', ArgFormats.ARTIFACT_VALUE),
    ])
    self.assertEqual(returned_values, set(['c']))

  def testArtifactFunctionParse(self):

    def func_a(
        examples: InputArtifact[standard_artifacts.Examples],
        model: OutputArtifact[standard_artifacts.Model],
        schema_uri: InputUri[standard_artifacts.Schema],
        statistics_uri: OutputUri[standard_artifacts.ExampleStatistics],
        num_steps: int
    ) -> OutputDict(
        precision=float, recall=float, message=Text, serialized_value=bytes):
      del examples, model, schema_uri, statistics_uri, num_steps
      return {
          'precision': 0.9,
          'recall': 0.8,
          'message': 'foo',
          'serialized_value': b'bar'
      }

    inputs, outputs, arg_formats, returned_values = (
        parse_typehint_component_function(func_a))
    self.assertEqual(
        inputs, {
            'examples': standard_artifacts.Examples,
            'schema_uri': standard_artifacts.Schema,
            'num_steps': standard_artifacts.Integer,
        })
    self.assertEqual(
        outputs, {
            'model': standard_artifacts.Model,
            'statistics_uri': standard_artifacts.ExampleStatistics,
            'precision': standard_artifacts.Float,
            'recall': standard_artifacts.Float,
            'message': standard_artifacts.String,
            'serialized_value': standard_artifacts.Bytes,
        })
    self.assertEqual(arg_formats, [
        ('examples', ArgFormats.INPUT_ARTIFACT),
        ('model', ArgFormats.OUTPUT_ARTIFACT),
        ('schema_uri', ArgFormats.INPUT_URI),
        ('statistics_uri', ArgFormats.OUTPUT_URI),
        ('num_steps', ArgFormats.ARTIFACT_VALUE),
    ])
    self.assertEqual(
        returned_values,
        set(['precision', 'recall', 'message', 'serialized_value']))

  def testEmptyReturnValue(self):
    # No output typehint.
    def func_a(examples: InputArtifact[standard_artifacts.Examples],
               model: OutputArtifact[standard_artifacts.Model], a: int,
               b: float):
      del examples, model, a, b

    # `None` output typehint.
    def func_b(examples: InputArtifact[standard_artifacts.Examples],
               model: OutputArtifact[standard_artifacts.Model], a: int,
               b: float) -> None:
      del examples, model, a, b

    # Both functions should be parsed in the same way.
    for func in [func_a, func_b]:
      inputs, outputs, arg_formats, returned_values = (
          parse_typehint_component_function(func))
      self.assertEqual(
          inputs, {
              'examples': standard_artifacts.Examples,
              'a': standard_artifacts.Integer,
              'b': standard_artifacts.Float,
          })
      self.assertEqual(outputs, {
          'model': standard_artifacts.Model,
      })
      self.assertEqual(arg_formats, [
          ('examples', ArgFormats.INPUT_ARTIFACT),
          ('model', ArgFormats.OUTPUT_ARTIFACT),
          ('a', ArgFormats.ARTIFACT_VALUE),
          ('b', ArgFormats.ARTIFACT_VALUE),
      ])
      self.assertEqual(returned_values, set([]))

  def testFunctionParseErrors(self):
    # Non-function arguments.
    with self.assertRaisesRegexp(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function(object())
    with self.assertRaisesRegexp(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function('foo')

    # Unannotated lambda.
    with self.assertRaisesRegexp(
        ValueError, 'must have all arguments annotated with typehints'):
      parse_typehint_component_function(lambda x: True)

    # Function with *args and **kwargs.
    with self.assertRaisesRegexp(
        ValueError,
        'must have either an OutputDict instance or None as its return'):

      def func_a(a: int, b: int) -> object:
        del a, b
        return object()

      parse_typehint_component_function(func_a)

    # Function with *args and **kwargs.
    with self.assertRaisesRegexp(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_b(a: int, b: int, *unused_args) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_b)
    with self.assertRaisesRegexp(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_c(a: int, b: int, **unused_kwargs) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_c)

    # Default arguments are not supported yet.
    # TODO(b/154332072): add support for default arguments.
    with self.assertRaisesRegexp(
        ValueError, 'currently does not support optional arguments'):

      def func_d(a: int, b: int = 1) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_d)

    # Not all arguments annotated with typehints.
    with self.assertRaisesRegexp(
        ValueError, 'must have all arguments annotated with typehint'):

      def func_e(a: int, b) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_e)

    # Artifact type used in annotation without `InputArtifact[ArtifactType]` or
    # `OutputArtifact[ArtifactType]` wrapper.
    with self.assertRaisesRegexp(
        ValueError, 'Invalid type hint annotation.*'
        'should indicate whether it is used as an input or output artifact'):

      def func_f(a: int,
                 unused_b: standard_artifacts.Examples) -> OutputDict(c=float):
        return {'c': float(a)}

      parse_typehint_component_function(func_f)

    # Invalid input typehint.
    with self.assertRaisesRegexp(ValueError, 'Unknown type hint annotation'):

      def func_g(a: int, b: Dict[int, int]) -> OutputDict(c=float):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_g)

    # Invalid output typehint.
    with self.assertRaisesRegexp(ValueError, 'Unknown type hint annotation'):

      def func_h(a: int, b: int) -> OutputDict(c='whatever'):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_h)

    # Output artifact in the wrong place.
    with self.assertRaisesRegexp(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_i(a: int, b: int) -> OutputDict(c=standard_artifacts.Examples):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_i)
    with self.assertRaisesRegexp(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_j(
          a: int,
          b: int) -> OutputDict(c=OutputArtifact[standard_artifacts.Examples]):
        return {'c': float(a + b)}

      parse_typehint_component_function(func_j)


if __name__ == '__main__':
  tf.test.main()
