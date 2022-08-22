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

import os
from typing import Dict, List, Optional, Any

import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.dsl.component.experimental.annotations import BeamComponentParameter
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import OutputDict
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import _SimpleBeamComponent
from tfx.dsl.component.experimental.decorators import _SimpleComponent
from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_dag_runner
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.system_executions import SystemExecution

_TestBeamPipelineArgs = ['--my_testing_beam_pipeline_args=foo']


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


class _InjectorAnnotation(SystemExecution):

  MLMD_SYSTEM_BASE_TYPE = 1


class _SimpleComponentAnnotation(SystemExecution):

  MLMD_SYSTEM_BASE_TYPE = 2


class _VerifyAnnotation(SystemExecution):

  MLMD_SYSTEM_BASE_TYPE = 3


@component
def _injector_1(
    foo: Parameter[int], bar: Parameter[str]) -> OutputDict(
        a=int, b=int, c=str, d=bytes):
  assert foo == 9
  assert bar == 'secret'
  return {'a': 10, 'b': 22, 'c': 'unicode', 'd': b'bytes'}


@component(component_annotation=_InjectorAnnotation)
def _injector_1_with_annotation(
    foo: Parameter[int], bar: Parameter[str]) -> OutputDict(
        a=int, b=int, c=str, d=bytes):
  assert foo == 9
  assert bar == 'secret'
  return {'a': 10, 'b': 22, 'c': 'unicode', 'd': b'bytes'}


@component
def _simple_component(
    a: int, b: int, c: str, d: bytes) -> OutputDict(
        e=float, f=float, g=Optional[str], h=Optional[str]):
  del c, d
  return {'e': float(a + b), 'f': float(a * b), 'g': 'OK', 'h': None}


@component(component_annotation=_SimpleComponentAnnotation)
def _simple_component_with_annotation(
    a: int, b: int, c: str, d: bytes) -> OutputDict(
        e=float, f=float, g=Optional[str], h=Optional[str]):
  del c, d
  return {'e': float(a + b), 'f': float(a * b), 'g': 'OK', 'h': None}


@component(use_beam=True)
def _simple_beam_component(
    a: int, b: int, c: str, d: bytes,
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
) -> OutputDict(
    e=float, f=float, g=Optional[str], h=Optional[str]):
  del c, d, beam_pipeline
  return {'e': float(a + b), 'f': float(a * b), 'g': 'OK', 'h': None}


def _verify_beam_pipeline_arg(a: int) -> OutputDict(b=float):
  return {'b': float(a)}


def _verify_beam_pipeline_arg_non_none_default_value(
    a: int,
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = beam.Pipeline()
) -> OutputDict(b=float):
  del beam_pipeline
  return {'b': float(a)}


@component
def _verify(e: float, f: float, g: Optional[str], h: Optional[str]):
  assert (e, f, g, h) == (32.0, 220.0, 'OK', None), (e, f, g, h)


@component(component_annotation=_VerifyAnnotation)
def _verify_with_annotation(e: float, f: float, g: Optional[str],
                            h: Optional[str]):
  assert (e, f, g, h) == (32.0, 220.0, 'OK', None), (e, f, g, h)


@component
def _injector_2(
    examples: OutputArtifact[standard_artifacts.Examples]
) -> OutputDict(
    a=int,
    b=float,
    c=str,
    d=bytes,
    e=str,
    f=List[Dict[str, float]],
    g=Dict[str, Dict[str, List[bool]]]):
  fileio.makedirs(examples.uri)
  return {
      'a': 1,
      'b': 2.0,
      'c': '3',
      'd': b'4',
      'e': 'passed',
      'f': [{
          'foo': 1.0
      }, {
          'bar': 2.0
      }],
      'g': {'foo': {'bar': [True, False]}}
  }


@component
def _injector_3(
    examples: OutputArtifact[standard_artifacts.Examples]
) -> OutputDict(
    a=int,
    b=float,
    c=str,
    d=bytes,
    e=str,
    f=Dict[str, Dict[str, List[bool]]],
    g=List[Dict[str, float]]):
  fileio.makedirs(examples.uri)
  return {
      'a': 1,
      'b': 2.0,
      'c': '3',
      'd': b'4',
      'e': None,
      'f': {'foo': {'bar': [True, False]}},
      'g': [{'foo': 1.0}, {'bar': 2.0}]
  }


@component
def _injector_4() -> OutputDict(
    a=Dict[str, List[List[Any]]],
    b=List[Any],
    c=Optional[Dict[str, Dict[str, Any]]],
    d=Dict[str, List[List[int]]],
    e=List[float],
    f=Dict[str, Dict[str, List[float]]]):
  return {
      'a': {'foo': [[1., 2]]},
      'b': [[{'e': 1}, {'e': 2}], [{'e': 3}, {'e': 4}]],
      'c': {'f': {'f': [1., 2.]}},
      'd': {'d': [[1, 2]]},
      'e': [1., 2.],
      'f': {'bar': {'bar': [1., 2.]}},
  }


@component
def _injector_4_invalid() -> OutputDict(
    a=Dict[str, List[List[int]]]):
  return {
      'a': {'foo': [[1.], [2]]},
  }


@component
def _json_compat_check_component(
    a: Optional[Dict[str, List[List[Any]]]] = None,
    b: Optional[List[Any]] = None,
    c: Optional[Dict[str, Dict[str, Any]]] = None,
    d: Optional[Dict[str, List[List[int]]]] = None,
    e: Optional[List[float]] = None,
    f: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
  del a, b, c, d, e, f


@component
def _optionalarg_component(
    foo: Parameter[int],
    bar: Parameter[str],
    examples: InputArtifact[standard_artifacts.Examples],
    a: int,
    b: float,
    c: str,
    d: bytes,
    e1: str = 'default',
    e2: Optional[str] = 'default',
    f: bytes = b'default',
    g: Parameter[float] = 1000.0,
    h: Parameter[str] = '2000',
    optional_examples_1: InputArtifact[standard_artifacts.Examples] = None,
    optional_examples_2: InputArtifact[standard_artifacts.Examples] = None,
    list_input: List[Dict[str, float]] = None,
    dict_input: Optional[Dict[str, Dict[str, List[bool]]]] = None,
    non_passed_dict: Dict[str, int] = None):
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
  # Test passed optional list input artifact.
  assert list_input == [{'foo': 1.0}, {'bar': 2.0}]
  # Test passed optional list input artifact.
  assert dict_input == {'foo': {'bar': [True, False]}}
  # Test non-passed optional list input artifact.
  assert non_passed_dict is None


@component(use_beam=True)
def _beam_component_with_artifact_inputs(
    foo: Parameter[int],
    a: int,
    b: float,
    c: str,
    d: bytes,
    examples: InputArtifact[standard_artifacts.Examples],
    processed_examples: OutputArtifact[standard_artifacts.Examples],
    dict_input: Dict[str, Dict[str, List[bool]]],
    e1: str = 'default',
    e2: Optional[str] = 'default',
    f: bytes = b'default',
    g: Parameter[float] = 1000.0,
    h: Parameter[str] = '2000',
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
    ):
  # Test non-optional parameters.
  assert foo == 9
  assert isinstance(examples, standard_artifacts.Examples)
  # Test non-optional `int`, `float`, `Text` and `bytes` input values.
  assert a == 1
  assert b == 2.0
  assert c == '3'
  assert d == b'4'
  assert dict_input == {'foo': {'bar': [True, False]}}
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
  del beam_pipeline
  fileio.makedirs(processed_examples.uri)


class ComponentDecoratorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
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

  def testSimpleBeamComponent(self):

    class _MySimpleBeamComponent(_SimpleBeamComponent):
      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(
          base_beam_executor.BaseBeamExecutor)

    input_channel = types.Channel(type=_InputArtifact)
    instance = _MySimpleBeamComponent(
        input=input_channel, folds=10).with_id(
            'my_instance').with_beam_pipeline_args(_TestBeamPipelineArgs)

    self.assertEqual(instance.executor_spec.beam_pipeline_args,
                     _TestBeamPipelineArgs)
    self.assertIs(instance.inputs['input'], input_channel)
    self.assertEqual(instance.outputs['output'].type, _OutputArtifact)
    self.assertEqual(instance.id, 'my_instance')

  def testDefinitionInClosureFails(self):
    with self.assertRaisesRegex(
        ValueError,
        'The @component decorator can only be applied to a function defined at '
        'the module level'):

      @component
      def my_component():  # pylint: disable=unused-variable
        return None

  def testNonKwargFails(self):
    with self.assertRaisesRegex(
        ValueError,
        'expects arguments to be passed as keyword arguments'):
      _injector_1(9, 'secret')

  def testNoBeamPipelineWhenUseBeamIsTrueFails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'The decorated function must have one and only one optional parameter '
        'of type BeamComponentParameter[beam.Pipeline] with '
        'default value None when use_beam=True.'):
      component(_verify_beam_pipeline_arg, use_beam=True)(a=1)

  def testBeamPipelineDefaultIsNotNoneFails(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'The default value for BeamComponentParameter must be None.'):
      component(
          _verify_beam_pipeline_arg_non_none_default_value, use_beam=True)(
              a=1)

  def testBeamExecutionSuccess(self):
    """Test execution with return values; success case."""
    instance_1 = _injector_1(foo=9, bar='secret')
    instance_2 = _simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    instance_3 = _verify(
        e=instance_2.outputs['e'],
        f=instance_2.outputs['f'],
        g=instance_2.outputs['g'],
        h=instance_2.outputs['h'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamComponentBeamExecutionSuccess(self):
    """Test execution with return values; success case."""
    instance_1 = _injector_1(foo=9, bar='secret')
    instance_2 = _simple_beam_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    instance_3 = _verify(
        e=instance_2.outputs['e'],
        f=instance_2.outputs['f'],
        g=instance_2.outputs['g'],
        h=instance_2.outputs['h'])  # pylint: disable=assignment-from-no-return

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
    instance_3 = _verify(
        e=instance_2.outputs['f'],
        f=instance_2.outputs['e'],
        g=instance_2.outputs['g'],
        h=instance_2.outputs['h'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    with self.assertRaisesRegex(
        RuntimeError, r'AssertionError: \(220.0, 32.0, \'OK\', None\)'):
      beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testOptionalInputsAndParameters(self):
    """Test execution with optional inputs and parameters."""
    instance_1 = _injector_2()  # pylint: disable=no-value-for-parameter
    self.assertLen(instance_1.outputs['examples'].get(), 1)
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
        optional_examples_1=instance_1.outputs['examples'],
        list_input=instance_1.outputs['f'],
        dict_input=instance_1.outputs['g'])

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamExecutionBeamComponentWithInputArtifactAndParameters(self):
    """Test execution of a beam component with InputArtifact and parameters."""
    instance_1 = _injector_2()  # pylint: disable=no-value-for-parameter
    self.assertLen(instance_1.outputs['examples'].get(), 1)
    instance_2 = _beam_component_with_artifact_inputs(  # pylint: disable=assignment-from-no-return, no-value-for-parameter
        foo=9,
        examples=instance_1.outputs['examples'],
        dict_input=instance_1.outputs['g'],
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'],
        e1=instance_1.outputs['e'],
        e2=instance_1.outputs['e'],
        g=999.0)

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testBeamExecutionNonNullableReturnError(self):
    """Test failure when None used for non-optional primitive return value."""
    instance_1 = _injector_3()  # pylint: disable=no-value-for-parameter
    self.assertLen(instance_1.outputs['examples'].get(), 1)
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
        optional_examples_1=instance_1.outputs['examples'],
        dict_input=instance_1.outputs['f'],
        list_input=instance_1.outputs['g'])

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])
    with self.assertRaisesRegex(
        ValueError, 'Non-nullable output \'e\' received None return value'):
      beam_dag_runner.BeamDagRunner().run(test_pipeline)

  def testComponentAnnotation(self):
    """Test component annotation parsed from decorator param."""
    instance_1 = _injector_1_with_annotation(foo=9, bar='secret')
    instance_2 = _simple_component_with_annotation(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])
    instance_3 = _verify_with_annotation(
        e=instance_2.outputs['e'],
        f=instance_2.outputs['f'],
        g=instance_2.outputs['g'],
        h=instance_2.outputs['h'])  # pylint: disable=assignment-from-no-return

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2, instance_3])

    beam_dag_runner.BeamDagRunner().run(test_pipeline)

    # Verify base_type annotation parsed from component decorator is correct.
    self.assertEqual(test_pipeline.components[0].type,
                     '__main__._injector_1_with_annotation')
    self.assertEqual(
        test_pipeline.components[0].type_annotation.MLMD_SYSTEM_BASE_TYPE, 1)
    self.assertEqual(test_pipeline.components[1].type,
                     '__main__._simple_component_with_annotation')
    self.assertEqual(
        test_pipeline.components[1].type_annotation.MLMD_SYSTEM_BASE_TYPE, 2)
    self.assertEqual(test_pipeline.components[2].type,
                     '__main__._verify_with_annotation')
    self.assertEqual(
        test_pipeline.components[2].type_annotation.MLMD_SYSTEM_BASE_TYPE, 3)

  def testJsonCompatible(self):
    instance_1 = _injector_4()
    instance_2 = _json_compat_check_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'],
        e=instance_1.outputs['e'],
        f=instance_1.outputs['f'],
    )
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])
    beam_dag_runner.BeamDagRunner().run(test_pipeline)

    instance_2 = _json_compat_check_component(
        a=instance_1.outputs['d'],
        b=instance_1.outputs['e'],
        c=instance_1.outputs['f'],
    )

    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[instance_1, instance_2])
    beam_dag_runner.BeamDagRunner().run(test_pipeline)

    for arg in ({'a': instance_1.outputs['b']},
                {'b': instance_1.outputs['c']},
                {'c': instance_1.outputs['d']},
                {'d': instance_1.outputs['e']},
                {'e': instance_1.outputs['f']},
                {'f': instance_1.outputs['a']},
                ):
      with self.assertRaisesRegex(
          TypeError, 'Argument.* should be a Channel of type .* \(got .*\)\.$'):  # pylint: disable=anomalous-backslash-in-string
        instance_2 = _json_compat_check_component(**arg)

    invalid_instance = _injector_4_invalid()
    instance_2 = _json_compat_check_component(
        a=invalid_instance.outputs['a'],
    )
    test_pipeline = pipeline.Pipeline(
        pipeline_name='test_pipeline_1',
        pipeline_root=self._test_dir,
        metadata_connection_config=metadata_config,
        components=[invalid_instance, instance_2])
    with self.assertRaisesRegex(
        TypeError,
        'Return value .* for output \'a\' is incompatible with output type .*$'
    ):
      beam_dag_runner.BeamDagRunner().run(test_pipeline)

if __name__ == '__main__':
  tf.test.main()
