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
"""Tests for tfx.dsl.components.base.decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Optional, Text
import unittest

# Standard Imports

import six
import tensorflow as tf
from tfx import types
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import _SimpleComponent
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.types import component_spec
from tfx.types import standard_artifacts


class _InputArtifact(types.Artifact):
  TYPE_NAME = '_InputArtifact'


class _OutputArtifact(types.Artifact):
  TYPE_NAME = '_OutputArtifact'


class _BasicComponentSpec(component_spec.ComponentSpec):

  PARAMETERS = {
      'folds': component_spec.ExecutionParameter(type=int),
  }
  INPUTS = {
      'input': component_spec.ChannelParameter(type=_InputArtifact),
  }
  OUTPUTS = {
      'output': component_spec.ChannelParameter(type=_OutputArtifact),
  }


if not six.PY2:
  # Currently, function components must be defined at the module level (not in
  # nested class or function scope). We define the test components here.

  @component
  def _injector_1(
      foo: Parameter[int], bar: Parameter[Text]) -> OutputDict(
          a=int, b=int, c=Text, d=bytes):
    assert foo == 9
    assert bar == 'secret'
    return {'a': 10, 'b': 22, 'c': 'unicode', 'd': b'bytes'}

  @component
  def _simple_component(a: int, b: int, c: Text, d: bytes) -> OutputDict(
      e=float, f=float):
    del c, d
    return {'e': float(a + b), 'f': float(a * b)}

  @component
  def _verify(e: float, f: float):
    assert (e, f) == (32.0, 220.0), (e, f)

  @component
  def _injector_2(
      examples: OutputArtifact[standard_artifacts.Examples]
  ) -> OutputDict(
      a=int, b=float, c=Text, d=bytes, e=Text):
    fileio.makedirs(examples.uri)
    return {'a': 1, 'b': 2.0, 'c': '3', 'd': b'4', 'e': 'passed'}

  @component
  def _optionalarg_component(
      foo: Parameter[int],
      bar: Parameter[Text],
      examples: InputArtifact[standard_artifacts.Examples],
      a: int,
      b: float,
      c: Text,
      d: bytes,
      e1: Text = 'default',
      e2: Optional[Text] = 'default',
      f: bytes = b'default',
      g: Parameter[float] = 1000.0,
      h: Parameter[Text] = '2000',
      optional_examples_1: InputArtifact[standard_artifacts.Examples] = None,
      optional_examples_2: InputArtifact[standard_artifacts.Examples] = None):
    # Test non-optional parameters.
    assert foo == 9
    assert bar == 'secret'
    assert isinstance(examples, standard_artifacts.Examples)
    # Test non-optional `int`, `float`, `Text` and `bytes` input values.
    assert a == 1
    assert b == 2.0
    assert c == '3'
    assert d == b'4'
    # Test passed optional arguments (with and without the `Optional` typehint
    # specifier).
    assert e1 == 'passed'
    assert e2 == 'passed'
    # Test that non-passed optional argument becomes the argument default.
    assert f == b'default'
    # Test passed optional parameter.
    assert g == 999.0
    # Test non-passed optional parameter.
    assert h == '2000'
    # Test passed optional input artifact.
    assert optional_examples_1 and optional_examples_1.uri
    # Test non-passed optional input artifact.
    assert optional_examples_2 is None


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class ComponentDecoratorTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentDecoratorTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._metadata_path = os.path.join(self._test_dir, 'metadata.db')

  def testSimpleComponent(self):

    class _MySimpleComponent(_SimpleComponent):
      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)

    input_channel = types.Channel(type=_InputArtifact)
    instance = _MySimpleComponent(
        input=input_channel, folds=10).with_id('my_instance')
    self.assertIs(instance.inputs['input'], input_channel)
    self.assertEqual(instance.outputs['output'].type, _OutputArtifact)
    self.assertEqual(instance.id, 'my_instance')

  def testDefinitionInClosureFails(self):
    with self.assertRaisesRegexp(
        ValueError,
        'The @component decorator can only be applied to a function defined at '
        'the module level'):

      @component
      def my_component():  # pylint: disable=unused-variable
        return None

  def testNonKwargFails(self):
    with self.assertRaisesRegexp(
        ValueError,
        'expects arguments to be passed as keyword arguments'):
      _injector_1(9, 'secret')

  def testBeamExecutionSuccess(self):
    """Test execution with return values; success case."""
    instance_1 = _injector_1(foo=9, bar='secret')
    instance_2 = _simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    instance_3 = _verify(e=instance_2.outputs['e'], f=instance_2.outputs['f'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamExecutionFailure(self):
    """Test execution with return values; failure case."""
    instance_1 = _injector_1(foo=9, bar='secret')
    instance_2 = _simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    # Swapped 'e' and 'f'.
    instance_3 = _verify(e=instance_2.outputs['f'], f=instance_2.outputs['e'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    with self.assertRaisesRegexp(RuntimeError,
                                 r'AssertionError: \(220.0, 32.0\)'):
      beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamExecutionOptionalInputsAndParameters(self):
    """Test execution with optional inputs and parameters."""
    instance_1 = _injector_2()  # pylint: disable=no-value-for-parameter
    self.assertEqual(1, len(instance_1.outputs['examples'].get()))
    instance_2 = _optionalarg_component(  # pylint: disable=assignment-from-no-return
        foo=9,
        bar='secret',
        examples=instance_1.outputs['examples'],
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'],
        e1=instance_1.outputs['e'],
        e2=instance_1.outputs['e'],
        g=999.0,
        optional_examples_1=instance_1.outputs['examples'])

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)


if __name__ == '__main__':
  tf.test.main()
