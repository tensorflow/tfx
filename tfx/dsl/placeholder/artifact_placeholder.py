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
"""Placeholder that evaluates to a component's input/output artifact."""

from __future__ import annotations

from typing import Any, Optional

from tfx.dsl.placeholder import placeholder_base
from tfx.proto.orchestration import placeholder_pb2

_types = placeholder_base.types


def input(key: str) -> ArtifactPlaceholder:  # pylint: disable=redefined-builtin
  """Returns a Placeholder that represents an input artifact.

  Args:
    key: The key of the input artifact.

  Returns:
    A Placeholder that supports

      1. Rendering the whole MLMD artifact proto as text_format.
         Example: `#!python input('model')`
      2. Accessing a specific index using `#!python [index]`, if multiple artifacts are
         associated with the given key. If not specified, default to the first
         artifact.
         Example: `#!python input('model')[0]`
      3. Getting the URI of an artifact through .uri property.
         Example: `#!python input('model').uri or input('model')[0].uri`
      4. Getting the URI of a specific split of an artifact using
         `#!python .split_uri(split_name)` method.
         Example: `#!python input('examples')[0].split_uri('train')`
      5. Getting the value of a primitive artifact through .value property.
         Example: `#!python input('primitive').value`
      6. Concatenating with other placeholders or strings.
         Example: `#!python input('model').uri + '/model/' + exec_property('version')`
  """
  return ArtifactPlaceholder(key, is_input=True)


def output(key: str) -> ArtifactPlaceholder:
  """Returns a Placeholder that represents an output artifact.

  It is the same as input(...) function, except it is for output artifacts.

  Args:
    key: The key of the output artifact.

  Returns:
    A Placeholder that supports

      1. Rendering the whole artifact as text_format.
         Example: `#!python output('model')`
      2. Accessing a specific index using [index], if multiple artifacts are
         associated with the given key. If not specified, default to the first
         artifact.
         Example: `#!python output('model')[0]`
      3. Getting the URI of an artifact through .uri property.
         Example: `#!python output('model').uri or output('model')[0].uri`
      4. Getting the URI of a specific split of an artifact using
         `#!python .split_uri(split_name)` method.
         Example: `#!python output('examples')[0].split_uri('train')`
      5. Getting the value of a primitive artifact through .value property.
         Example: `#!python output('primitive').value`
      6. Concatenating with other placeholders or strings.
         Example: `#!python output('model').uri + '/model/' + exec_property('version')`
  """
  return ArtifactPlaceholder(key, is_input=False)


class ArtifactPlaceholder(placeholder_base.Placeholder):
  """Artifact Placeholder represents an input or an output artifact.

  Prefer to use ph.input(...) or ph.output(...) to create instances.
  """

  def __init__(
      self,
      key: str,
      is_input: bool,
      index: Optional[int] = None,
  ):
    """Initializes the class. Consider this private."""
    # This should be tfx.types.Artifact, but it can't be due to a circular
    # dependency. See placeholder_base.py for details. TODO(b/191610358).
    super().__init__(expected_type=None)
    assert index is None or isinstance(index, int)
    self._key = key
    self._is_input = is_input
    self._index = index

  @property
  def is_input(self) -> bool:
    return self._is_input

  @property
  def is_output(self) -> bool:
    return not self._is_input

  @property
  def key(self) -> str:
    return self._key

  @property
  def uri(self) -> _ArtifactUriOperator:
    return _ArtifactUriOperator(self)

  def split_uri(self, split: str) -> _ArtifactUriOperator:
    return _ArtifactUriOperator(self, split)

  @property
  def value(self) -> _ArtifactValueOperator:
    if self.is_output:
      raise ValueError('Calling ph.output(..).value is not supported.')
    return _ArtifactValueOperator(self)

  def __getitem__(self, index: int) -> ArtifactPlaceholder:
    assert self._index is None
    return ArtifactPlaceholder(self._key, self._is_input, index)

  def property(self, key: str) -> _PropertyOperator:
    return _PropertyOperator(self, key)

  def custom_property(self, key: str) -> _PropertyOperator:
    return _PropertyOperator(self, key, is_custom_property=True)

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return (
        isinstance(other, ArtifactPlaceholder)
        and self._key == other._key  # pylint: disable=protected-access
        and self._is_input == other._is_input  # pylint: disable=protected-access
        and self._index == other._index  # pylint: disable=protected-access
    )

  def encode(
      self, component_spec: Any = None
  ) -> placeholder_pb2.PlaceholderExpression:
    del component_spec
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.index_op.index = self._index or 0
    artifact_result = result.operator.index_op.expression
    artifact_result.placeholder.type = (
        placeholder_pb2.Placeholder.INPUT_ARTIFACT
        if self._is_input
        else placeholder_pb2.Placeholder.OUTPUT_ARTIFACT
    )
    if self._key:
      artifact_result.placeholder.key = self._key
    return result


class _ArtifactUriOperator(placeholder_base.UnaryPlaceholderOperator):
  """Artifact URI Operator extracts the URI from an artifact Placeholder.

  Prefer to use the .uri property of ArtifactPlaceholder.
  """

  def __init__(self, value: placeholder_base.Placeholder, split: str = ''):
    super().__init__(value, expected_type=str)
    self._split = split

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return (
        isinstance(other, _ArtifactUriOperator)
        and self._split == other._split  # pylint: disable=protected-access
        and super().internal_equals(other)
    )

  def encode(
      self, component_spec: Optional[type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_uri_op.expression.CopyFrom(
        self._value.encode(component_spec)
    )
    if self._split:
      result.operator.artifact_uri_op.split = self._split
    return result


class _ArtifactValueOperator(placeholder_base.UnaryPlaceholderOperator):
  """Artifact Value Operator extracts the value from a primitive artifact Placeholder.

  Prefer to use the .value property of ArtifactPlaceholder.
  """

  def __init__(self, value: placeholder_base.Placeholder, split: str = ''):
    super().__init__(value, expected_type=placeholder_base.ValueType)
    self._split = split

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return (
        isinstance(other, _ArtifactValueOperator)
        and self._split == other._split  # pylint: disable=protected-access
        and super().internal_equals(other)
    )

  def encode(
      self, component_spec: Optional[type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_value_op.expression.CopyFrom(
        self._value.encode(component_spec)
    )
    return result


class _PropertyOperator(placeholder_base.UnaryPlaceholderOperator):
  """Property Operator gets the property of an artifact Placeholder.

  Prefer to use .property(key) method of Artifact Placeholder.
  """

  def __init__(
      self,
      value: placeholder_base.Placeholder,
      key: str,
      is_custom_property: bool = False,
  ):
    super().__init__(value, expected_type=placeholder_base.ValueType)
    self._key = key
    self._is_custom_property = is_custom_property

  def internal_equals(self, other: placeholder_base.Placeholder) -> bool:
    return (
        isinstance(other, _PropertyOperator)
        and self._key == other._key  # pylint: disable=protected-access
        and self._is_custom_property == other._is_custom_property  # pylint: disable=protected-access
        and super().internal_equals(other)
    )

  def encode(
      self, component_spec: Optional[type['_types.ComponentSpec']] = None
  ) -> placeholder_pb2.PlaceholderExpression:
    result = placeholder_pb2.PlaceholderExpression()
    result.operator.artifact_property_op.expression.CopyFrom(
        self._value.encode(component_spec)
    )
    result.operator.artifact_property_op.key = self._key
    result.operator.artifact_property_op.is_custom_property = (
        self._is_custom_property
    )
    return result
