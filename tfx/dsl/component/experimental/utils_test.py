# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.component.experimental.utils."""


import copy
import inspect
from typing import Dict, List
import tensorflow as tf
from tfx.dsl.component.experimental import annotations
from tfx.dsl.component.experimental import annotations_test_proto_pb2
from tfx.dsl.component.experimental import decorators
from tfx.dsl.component.experimental import function_parser
from tfx.dsl.component.experimental import utils
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import system_executions


def top_level_func() -> None:
  pass


def _private_func() -> None:
  pass


class UtilsTest(tf.test.TestCase):
  # pylint: disable=g-error-prone-assert-raises
  # pylint: disable=unused-argument

  def test_assert_is_func_type_succeeds(self):
    def func() -> str:
      return 'foo'

    utils.assert_is_functype(func)

  def test_assert_no_private_func_in_main_succeeds(self):
    _private_func.__module__ = '__main__'

    with self.assertRaisesRegex(
        ValueError,
        r'Custom Python functions \(both @component and pre/post hooks\)',
    ):
      utils.assert_no_private_func_in_main(_private_func)

  def test_assert_is_func_type_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected a typehint-annotated Python function'
    ):
      utils.assert_is_functype('not_func')

  def test_assert_no_varargs_varkw_succeeds(self):
    def func_with_two_params(param1: str, param2: int) -> None:
      pass

    utils.assert_no_varargs_varkw(
        inspect.getfullargspec(func_with_two_params),
    )

  def test_assert_no_varargs_varkw_raises_error(self):
    def func_with_varargs(*args) -> None:
      pass

    def func_with_varkw(**kwargs) -> None:
      pass

    with self.assertRaisesRegex(
        ValueError, r'does not support \*args or \*\*kwargs'
    ):
      utils.assert_no_varargs_varkw(
          inspect.getfullargspec(func_with_varargs),
      )

    with self.assertRaisesRegex(
        ValueError, r'does not support \*args or \*\*kwargs'
    ):
      utils.assert_no_varargs_varkw(
          inspect.getfullargspec(func_with_varkw),
      )

  def test_extract_arg_defaults(self):
    def func_with_default_values(
        param1: str, param2: int = 2, param3: str = 'foo'
    ) -> None:
      pass

    arg_defaults = utils.extract_arg_defaults(
        inspect.getfullargspec(func_with_default_values),
    )
    self.assertEqual(arg_defaults, {'param2': 2, 'param3': 'foo'})

  def test_parse_parameter_arg_succeeds(self):

    def func_with_primitive_parameter(
        foo: annotations.Parameter[int],
        int_param: annotations.Parameter[int],
        float_param: annotations.Parameter[float],
        str_param: annotations.Parameter[str],
        bool_param: annotations.Parameter[bool],
        proto_param: annotations.Parameter[
            annotations_test_proto_pb2.TestMessage
        ],
        dict_int_param: annotations.Parameter[Dict[str, int]],
        list_bool_param: annotations.Parameter[List[bool]],
        dict_list_bool_param: annotations.Parameter[Dict[str, List[bool]]],
        list_dict_float_param: annotations.Parameter[List[Dict[str, float]]],
    ) -> None:
      pass

    arg_defaults = utils.extract_arg_defaults(
        inspect.getfullargspec(func_with_primitive_parameter),
    )
    typehints = func_with_primitive_parameter.__annotations__
    original_parameters = {'foo': int}
    original_arg_formats = {'foo': utils.ArgFormats.PARAMETER}
    expected_arg_types_by_name = {
        'int_param': int,
        'float_param': float,
        'str_param': str,
        'bool_param': bool,
        'proto_param': annotations_test_proto_pb2.TestMessage,
        'dict_int_param': Dict[str, int],
        'list_bool_param': List[bool],
        'dict_list_bool_param': Dict[str, List[bool]],
        'list_dict_float_param': List[Dict[str, float]],
    }
    for arg_name, arg_type in expected_arg_types_by_name.items():
      parameters = copy.deepcopy(original_parameters)
      arg_formats = copy.deepcopy(original_arg_formats)
      utils.parse_parameter_arg(
          arg_name,
          arg_defaults,
          typehints[arg_name],
          parameters,
          arg_formats,
      )
      self.assertEqual(parameters, {'foo': int, arg_name: arg_type})
      self.assertEqual(
          arg_formats,
          {
              'foo': utils.ArgFormats.PARAMETER,
              arg_name: utils.ArgFormats.PARAMETER,
          },
      )

  def test_parse_parameter_arg_raises_error_mismatched_types(self):
    def func_mismatched_types(
        int_param: annotations.Parameter[int] = 'foo',
    ) -> None:
      pass

    arg_defaults = utils.extract_arg_defaults(
        inspect.getfullargspec(func_mismatched_types),
    )
    typehints = func_mismatched_types.__annotations__
    with self.assertRaisesRegex(
        ValueError, 'must be an instance of its declared type .* or `None`'
    ):
      utils.parse_parameter_arg(
          'int_param',
          arg_defaults,
          typehints['int_param'],
          {},
          {},
      )

  def test_assert_is_top_level_func_succeeds(self):
    utils.assert_is_top_level_func(top_level_func)

  def test_assert_is_top_level_func_raises_error(self):
    def nested_func() -> None:
      pass

    with self.assertRaisesRegex(
        ValueError,
        'can only be applied to a function defined at the module level',
    ):
      utils.assert_is_top_level_func(nested_func)

  def test_create_component_class(self):
    # pytype: disable=invalid-annotation
    def func(
        primitive_bool_input: bool,
        artifact_int_input: annotations.InputArtifact[
            standard_artifacts.Integer
        ],
        artifact_examples_output: annotations.OutputArtifact[
            standard_artifacts.Examples
        ],
        int_param: annotations.Parameter[int],
        proto_param: annotations.Parameter[
            annotations_test_proto_pb2.TestMessage
        ],
        json_compat_param: annotations.Parameter[Dict[str, int]],
        str_param: annotations.Parameter[str] = 'foo',
    ) -> annotations.OutputDict(
        str_output=str,
        str_list_output=List[str],
        map_str_float_output=Dict[str, float],
    ):
      pass

    # pytype: enable=invalid-annotation

    (
        inputs,
        outputs,
        parameters,
        arg_formats,
        arg_defaults,
        returned_values,
        json_typehints,
        return_json_typehints,
    ) = function_parser.parse_typehint_component_function(func)
    type_annotation = system_executions.Process
    base_executor_class = decorators._FunctionExecutor
    executor_spec_class = executor_spec.ExecutorClassSpec
    base_component_class = decorators._SimpleComponent
    actual_component_class = utils.create_component_class(
        func=func,
        arg_defaults=arg_defaults,
        arg_formats=arg_formats,
        base_executor_class=base_executor_class,
        executor_spec_class=executor_spec_class,
        base_component_class=base_component_class,
        inputs=inputs,
        outputs=outputs,
        parameters=parameters,
        type_annotation=type_annotation,
        json_compatible_inputs=json_typehints,
        json_compatible_outputs=return_json_typehints,
        return_values_optionality=returned_values,
    )

    actual_spec_class = actual_component_class.SPEC_CLASS
    spec_inputs = actual_spec_class.INPUTS
    self.assertLen(spec_inputs, 2)
    self.assertEqual(
        spec_inputs['primitive_bool_input'].type, standard_artifacts.Boolean
    )
    self.assertEqual(
        spec_inputs['artifact_int_input'].type, standard_artifacts.Integer
    )
    spec_outputs = actual_spec_class.OUTPUTS
    self.assertLen(spec_outputs, 4)
    self.assertEqual(
        spec_outputs['artifact_examples_output'].type,
        standard_artifacts.Examples,
    )
    self.assertEqual(spec_outputs['str_output'].type, standard_artifacts.String)
    self.assertEqual(
        spec_outputs['str_list_output'].type, standard_artifacts.JsonValue
    )
    self.assertEqual(
        spec_outputs['map_str_float_output'].type, standard_artifacts.JsonValue
    )
    spec_parameter = actual_spec_class.PARAMETERS
    self.assertLen(spec_parameter, 4)
    self.assertEqual(spec_parameter['int_param'].type, int)
    self.assertEqual(spec_parameter['int_param'].optional, False)
    self.assertEqual(spec_parameter['str_param'].type, str)
    self.assertEqual(spec_parameter['str_param'].optional, True)
    self.assertEqual(
        spec_parameter['proto_param'].type,
        annotations_test_proto_pb2.TestMessage,
    )
    self.assertEqual(spec_parameter['json_compat_param'].type, Dict[str, int])
    self.assertEqual(spec_parameter['json_compat_param'].optional, False)
    self.assertEqual(actual_spec_class.TYPE_ANNOTATION, type_annotation)

    self.assertIsInstance(
        actual_component_class.EXECUTOR_SPEC, executor_spec_class
    )
    executor_class = actual_component_class.EXECUTOR_SPEC.executor_class
    self.assertEqual(executor_class._ARG_FORMATS, arg_formats)
    self.assertEqual(executor_class._ARG_DEFAULTS, arg_defaults)
    self.assertEqual(executor_class._FUNCTION, func)
    self.assertEqual(executor_class._RETURNED_VALUES, returned_values)
    self.assertEqual(
        executor_class._RETURN_JSON_COMPAT_TYPEHINT, return_json_typehints
    )
    self.assertEqual(executor_class.__module__, func.__module__)
    self.assertIsInstance(executor_class, type(base_executor_class))

    self.assertIsInstance(actual_component_class, type(base_component_class))
    self.assertEqual(actual_component_class.__module__, func.__module__)
    self.assertEqual(actual_component_class.test_call, func)  # pytype: disable=attribute-error
