# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.component.experimental.component_utils."""

from typing import Any, Callable, Optional

import tensorflow as tf
from tfx.dsl.component.experimental import component_utils
from tfx.types import component_spec
from tfx.types import standard_artifacts


_ExecutionParameter = component_spec.ExecutionParameter
_ChannelParameter = component_spec.ChannelParameter

_Model = standard_artifacts.Model
_Examples = standard_artifacts.Examples
_Integer = standard_artifacts.Integer
_JsonValue = standard_artifacts.JsonValue


class TestComponentSpec(component_spec.ComponentSpec):
  PARAMETERS = {
      'param_int': _ExecutionParameter(int),
      'param_map': _ExecutionParameter(dict[str, int]),
      'param_optional_int': _ExecutionParameter(int, optional=True),
  }
  INPUTS = {
      'input_model': _ChannelParameter(_Model),
      'input_optional_model': _ChannelParameter(_Model, optional=True),
      'input_integer': _ChannelParameter(_Integer),
      'input_optional_integer': _ChannelParameter(_Integer, optional=True),
      'input_json': _ChannelParameter(_JsonValue),
      'input_optional_json': _ChannelParameter(_JsonValue, optional=True),
  }
  OUTPUTS = {
      'output_model': _ChannelParameter(_Model),
      'output_integer': _ChannelParameter(_Integer),
      'output_json': _ChannelParameter(_JsonValue),
  }


class IAmNotJsonable:
  pass


class ComponentUtilsTest(tf.test.TestCase):

  def _assert_type_check_execution_function_params_ok(
      self, execution: Callable[..., Any]
  ):
    component_utils._type_check_execution_function_params(
        TestComponentSpec, execution
    )

  def _assert_type_check_execution_function_params_error(
      self,
      execution: Callable[..., Any],
      expected_error_type: type[Exception] = TypeError,
  ):
    with self.assertRaises(expected_error_type):
      component_utils._type_check_execution_function_params(
          TestComponentSpec, execution
      )

  def test_type_check_valid_types(self):
    def execution(
        param_int: int,
        param_map: dict[str, int],
        param_optional_int: Optional[int],
        input_model: _Model,
        input_optional_model: Optional[_Model],
        input_integer: int,
        input_optional_integer: Optional[int],
        input_json: dict[str, int],
        input_optional_json: Optional[dict[str, int]],
        output_model: _Model,
    ):
      del (
          param_int,
          param_map,
          param_optional_int,
          input_model,
          input_optional_model,
          input_integer,
          input_optional_integer,
          input_json,
          input_optional_json,
          output_model,
      )

    self._assert_type_check_execution_function_params_ok(execution)

  def test_type_check_valid_list_artifacts(self):

    def execution(
        input_model: list[_Model],
        output_model: list[_Model],
    ):
      del input_model, output_model

    self._assert_type_check_execution_function_params_ok(execution)

  def test_type_check_raises_error_invalid_types(self):

    def execution_param_int(param_int: str):
      del param_int

    def execution_param_map(param_map: str):
      del param_map

    def execution_input_model(input_model: _Examples):
      del input_model

    def execution_input_integer(input_integer: str):
      del input_integer

    def execution_output_model(output_model: _Examples):
      del output_model

    self._assert_type_check_execution_function_params_error(execution_param_int)
    self._assert_type_check_execution_function_params_error(execution_param_map)
    self._assert_type_check_execution_function_params_error(
        execution_input_model
    )
    self._assert_type_check_execution_function_params_error(
        execution_input_integer
    )
    self._assert_type_check_execution_function_params_error(
        execution_output_model
    )

  def test_type_check_raises_error_primitive_type_param_for_outputs(self):

    def execution_output_int(output_integer: int):
      del output_integer

    def execution_output_json(output_json: dict[str, int]):
      del output_json

    self._assert_type_check_execution_function_params_error(
        execution_output_int
    )
    self._assert_type_check_execution_function_params_error(
        execution_output_json
    )

  def test_type_check_raises_error_for_misaligned_optionalities(
      self,
  ):

    def execution_param_1(param_int: Optional[int]):
      del param_int

    def execution_param_2(param_optional_int: int):
      del param_optional_int

    def execution_model_1(input_model: Optional[_Model]):
      del input_model

    def execution_model_2(input_optional_model: _Model):
      del input_optional_model

    def execution_integer_1(input_integer: Optional[int]):
      del input_integer

    def execution_integer_2(input_optional_integer: int):
      del input_optional_integer

    def execution_json_1(input_json: Optional[dict[str, int]]):
      del input_json

    def execution_json_2(input_optional_json: dict[str, int]):
      del input_optional_json

    self._assert_type_check_execution_function_params_error(execution_param_1)
    self._assert_type_check_execution_function_params_error(execution_param_2)
    self._assert_type_check_execution_function_params_error(execution_model_1)
    self._assert_type_check_execution_function_params_error(execution_model_2)
    self._assert_type_check_execution_function_params_error(execution_integer_1)
    self._assert_type_check_execution_function_params_error(execution_integer_2)
    self._assert_type_check_execution_function_params_error(execution_json_1)
    self._assert_type_check_execution_function_params_error(execution_json_2)

  def test_type_check_raises_error_not_jsonable_param_type(self):

    def execution(input_json: IAmNotJsonable):
      del input_json

    def execution_optional(input_optional_json: Optional[IAmNotJsonable]):
      del input_optional_json

    self._assert_type_check_execution_function_params_error(execution)
    self._assert_type_check_execution_function_params_error(execution_optional)

  def test_type_check_raises_error_no_matching_name_from_spec(self):

    def execution(invalid_name: int):
      del invalid_name

    self._assert_type_check_execution_function_params_error(
        execution, expected_error_type=AttributeError
    )
