# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.typing_utils."""

import sys
import typing
from typing import Any, List, Literal, TypedDict
import unittest

import tensorflow as tf
import tfx.types
from tfx.types import standard_artifacts
from tfx.utils import typing_utils
import typing_extensions

import ml_metadata as mlmd


class TypingUtilsTest(tf.test.TestCase):

  def _model(self):
    return standard_artifacts.Model()

  def test_artifact_multimap_pylint(self):

    def get_artifact(input_dict: typing_utils.ArtifactMultiMap):
      return input_dict['model'][0]

    def get_type_name(artifact: tfx.types.Artifact):
      return artifact.type_name

    # No pytype complain
    input_dict = {'model': [self._model()]}
    self.assertEqual(get_type_name(get_artifact(input_dict)), 'Model')

  def test_is_artifact_multimap(self):

    def yes(value: Any):
      self.assertTrue(typing_utils.is_artifact_multimap(value))

    def no(value: Any):
      self.assertFalse(typing_utils.is_artifact_multimap(value))

    yes({})
    yes({'model': []})
    yes({'model': [self._model()]})
    yes({'model': [self._model(), self._model()]})
    no({'model': [self._model(), 'not an artifact']})
    no({'model': self._model()})
    no({123: [self._model()]})

  def test_is_list_of_artifact_multimap(self):

    def yes(value: Any):
      self.assertTrue(typing_utils.is_list_of_artifact_multimap(value))

    def no(value: Any):
      self.assertFalse(typing_utils.is_list_of_artifact_multimap(value))

    yes([])
    yes([{}])
    yes([{}, {}])
    yes([{'model': []}])
    yes([{'model': [self._model()]}])
    yes([{'model': [self._model(), self._model()]}])
    no([self._model()])
    no([{'model': self._model()}])

  def assertIsCompatible(self, value: Any, tp: Any):
    with self.subTest(f'{value} is compatible with {tp}'):
      self.assertTrue(typing_utils.is_compatible(value, tp))

  def assertIsNotCompatible(self, value: Any, tp: Any):
    with self.subTest(f'{value} is NOT compatible with {tp}'):
      self.assertFalse(typing_utils.is_compatible(value, tp))

  def test_is_compatible(self):
    yes = self.assertIsCompatible
    no = self.assertIsNotCompatible

    # Any
    yes(0, typing.Any)
    yes(0.0, typing.Any)
    yes('a', typing.Any)
    yes(False, typing.Any)
    yes([], typing.Any)
    yes((), typing.Any)
    yes({}, typing.Any)

    # Primitives
    yes(0, int)
    no(0, str)
    yes(0.0, float)
    no(0.0, int)
    yes(False, bool)
    yes(False, int)
    yes('a', str)
    no('a', int)

    # Non-types
    yes(None, None)
    yes(None, type(None))
    no(0, None)

    # Lists
    yes([0], list)
    yes([0], typing.List)
    yes([0], typing.List[int])
    yes([0], typing.Iterable)
    yes([0], typing.Iterable[int])
    yes([0], typing.Sequence[int])
    yes([0], typing.MutableSequence[int])
    yes([], typing.List[int])  # Empty list
    no([0], typing.List[str])
    no(['a', 0], typing.List[str])  # Heterogeneous list
    yes([[0.0, 1.0], [1.0, 0.0]], typing.List[typing.List[float]])

    # Sets
    yes({0}, set)
    yes({0}, typing.Set)
    yes({0}, typing.Set[int])
    no({0}, typing.Set[float])

    # Tuples
    yes((), tuple)
    yes((), typing.Tuple)
    yes(('a', 0), typing.Tuple)
    no(['a', 0], typing.Tuple)
    yes((0, 1, 2, 3), typing.Tuple[int, ...])  # Ellipsis
    no((0, 1, 2, 'a'), typing.Tuple[int, ...])  # Ellipsis
    yes((0, 'a', ()), typing.Tuple[int, str, tuple])

    # Dictionaries
    yes({'a': 0}, dict)
    yes({'a': 0}, typing.Dict)
    yes({'a': 0}, typing.Dict[str, int])
    no([], typing.Dict[str, str])
    no({'a': 0}, typing.Dict[int, int])  # Wrong key type
    no({'a': 0}, typing.Dict[str, str])  # Wrong value type
    yes({'a': [0, 1, 2, 3]}, typing.Dict[str, typing.List[int]])
    yes({'a': [0, 1, 2, 3]}, typing.Mapping[str, typing.Sequence[int]])
    no({'a': [0, 'a']}, typing.Dict[str, typing.List[int]])
    no({'a': 0, 'b': 0.0}, typing.Dict[str, int])
    no({'a': 0, 0: 'a'}, typing.Dict[str, int])
    yes({'a': 0, 0: 'a'}, typing.Dict[
        typing.Union[str, int], typing.Union[str, int]])

    # Types
    class Foo:
      pass

    class SubFoo(Foo):
      pass

    yes(int, type)
    yes(int, typing.Type[int])
    yes(bool, typing.Type[int])
    yes(SubFoo, typing.Type[Foo])
    no(int, typing.Type[Foo])
    yes(str, typing.Type[typing.Union[str, int]])
    yes(int, typing.Type[typing.Union[str, int]])
    no(float, typing.Type[typing.Union[str, int]])

    # Unions
    long_union = typing.Union[int, str, typing.Union[tuple, typing.List[int]]]
    yes(0, long_union)
    yes('a', long_union)
    yes(('a', 0), long_union)
    yes([0, 1], long_union)
    no(0.0, long_union)
    no([0, 'a'], long_union)

    # Optional
    yes(0, typing.Optional[int])
    yes(None, typing.Optional[int])
    yes([0], typing.Optional[typing.List[int]])
    yes(None, typing.Optional[typing.List[int]])
    yes({'a': 0}, typing.Optional[typing.Dict[str, int]])
    yes(None, typing.Optional[typing.Dict[str, int]])

    # Literal
    literal = Literal['a', 'b', 'c']
    yes('a', literal)
    yes('b', literal)
    yes('c', literal)
    no('d', literal)
    no(0, literal)

  def test_is_compatible_typed_dict_total(self):
    class Total(TypedDict, total=True):
      x: int
      y: float
      z: str

    self.assertIsCompatible({'x': 1, 'y': 0.4, 'z': 'foo'}, Total)
    self.assertIsNotCompatible({'x': '1', 'y': 0.4, 'z': 'foo'}, Total)
    self.assertIsNotCompatible({'x': 1, 'z': 'foo'}, Total)
    self.assertIsNotCompatible(
        {'w': False, 'x': 1, 'y': 0.4, 'z': 'foo'}, Total
    )

  def test_is_compatible_typed_dict_not_total(self):
    class NotTotal(TypedDict, total=False):
      x: int
      y: float
      z: str

    self.assertIsCompatible({'x': 1, 'y': 0.4, 'z': 'foo'}, NotTotal)
    self.assertIsNotCompatible({'x': '1', 'y': 0.4, 'z': 'foo'}, NotTotal)
    self.assertIsCompatible({'x': 1, 'z': 'foo'}, NotTotal)
    self.assertIsCompatible({}, NotTotal)
    self.assertIsNotCompatible({'w': False}, NotTotal)

  def test_is_compatible_typed_dict_nested(self):
    class Inner(TypedDict):
      x: int

    class Outer(TypedDict):
      x: Inner  # pytype: disable=invalid-annotation

    self.assertIsCompatible({'x': {'x': 42}}, Outer)
    self.assertIsNotCompatible({'x': {}}, Outer)
    self.assertIsNotCompatible({'x': 42}, Outer)

  def test_is_compatible_typed_dict_partial(self):
    class Partial(TypedDict):
      x: int
      y: typing_extensions.NotRequired[int]  # type: ignore

    self.assertIsCompatible({'x': 1, 'y': 2}, Partial)
    if sys.version_info >= (3, 11, 0):
      self.assertIsCompatible({'x': 1}, Partial)
    self.assertIsNotCompatible({}, Partial)
    self.assertIsNotCompatible({'z': 3}, Partial)

  # pytype: disable=not-supported-yet
  # NOTE: Sadly, these typing_extensions.assert_type() don't work today, i.e.
  # they don't actually detect bad types. If they still don't work in the Python
  # 3.10 future, we should switch to pytype_extensions.assert_type() once the
  # version in dependencies.py is 4.2+.
  @unittest.skipUnless(
      hasattr(typing_extensions, 'assert_type'),
      'Prevent failure on older Python versions',
  )
  def test_is_compatible_narrowing(self):
    value = 'foo'
    typing_extensions.assert_type(value, str)
    if typing_utils.is_compatible(value, int):
      typing_extensions.assert_type(value, int)
    elif typing_utils.is_compatible(value, List[tfx.types.Artifact]):
      typing_extensions.assert_type(value, List[tfx.types.Artifact])

  # pytype: enable=not-supported-yet

  def test_str_is_not_compatible_with_iterable_str(self):
    self.assertIsNotCompatible('abc', typing.Iterable[str])
    self.assertIsCompatible(['abc'], typing.Iterable[str])
    self.assertIsNotCompatible('abc', typing.Sequence[str])
    self.assertIsNotCompatible('abc', typing.Collection[str])
    self.assertIsNotCompatible('abc', typing.Container[str])
    # Should explicitly allow union of str or Iterable[Any].
    # https://github.com/google/pytype/blob/main/docs/faq.md#why-doesnt-str-match-against-string-iterables
    self.assertIsCompatible('abc', typing.Union[typing.Iterable[str], str])
    self.assertIsCompatible('abc', typing.Iterable[Any])

  def test_is_compatible_proto_enum(self):
    State = mlmd.proto.Artifact.State  # pylint: disable=invalid-name

    self.assertIsCompatible(State.UNKNOWN, State)
    self.assertIsCompatible(State.PENDING, State)
    self.assertIsCompatible(State.LIVE, State)
    self.assertIsCompatible(0, State)
    self.assertIsCompatible(1, State)
    self.assertIsCompatible(2, State)

    self.assertIsNotCompatible(-1, State)  # Out of range.
    self.assertIsNotCompatible(999, State)  # Out of range.
    self.assertIsNotCompatible('LIVE', State)  # String name doesn't count.


if __name__ == '__main__':
  tf.test.main()
