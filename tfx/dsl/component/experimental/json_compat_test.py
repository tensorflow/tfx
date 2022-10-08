# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.component.experimental.json_compat."""

from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tfx.dsl.component.experimental.json_compat import check_strict_json_compat
from tfx.dsl.component.experimental.json_compat import is_json_compatible


class JsonCompatTest(tf.test.TestCase):

  def testIsJsonCompatible(self):
    for typehint in (
        Dict[str, float], List[int], Dict[str, List[Dict[str, int]]],
        Optional[Dict[str, Dict[str, bool]]],
        Optional[Dict[str, Optional[List[int]]]],
        Union[Dict[str, int], type(None)], Union[Dict[str, str], List[bool]],
        Dict[str, Any], Dict[str, List[Any]], List[Any], List[Dict[str, Any]]):
      self.assertTrue(is_json_compatible(typehint))
    for typehint in (
        # Bare primitives.
        dict, Dict, Union,
        # Invalid Dict, Union or List parameters.
        Dict[str, Dict], Dict[str, bytes], Dict[int, float],
        Union[Dict[str, int], float], List[bytes], List['Y'],
        # Primitive types.
        int, str, float, dict, bytes, bool, type(None), Any):
      self.assertFalse(is_json_compatible(typehint))

  def testCheckStrictJsonCompat(self):
    for pair in (
        # Pairs are currently valid but not supported by is_json_compatible.
        (int, int), (List, List), (Dict, Dict), (List[str], List),
        (Dict[str, int], Dict), (Dict[str, float], Dict[str, float]),
        # Valid pairs are supported by is_json_compatible.
        (List[bool], List[bool]),
        (Dict[str, int], Union[Dict[str, int], List[int]]),
        (Dict[str, bool], Dict[str, Any]), (List[Dict[str, int]], List[Any]),
        (Dict[str, List[str]], Dict[str, Any]),
        (List[Dict[str, str]], List[Any]), (List[Any], List[Any]),
        (Dict[str, Any], Dict[str, Any])):
      self.assertTrue(check_strict_json_compat(pair[0], pair[1]))

    for pair in (
        (str, int), (Dict[str, int], Dict[str, str]),
        (Dict, Dict[str, int]), (List, List[float]), (type(None), int),
        (Dict[str, Any], Dict[str, str]),
        (Dict[str, str], List[Any]),
        (List[Any], List[Dict[str, bool]]), (List[Any], Dict[str, Any])):
      self.assertFalse(check_strict_json_compat(pair[0], pair[1]))

    # Runtime type check.
    self.assertTrue(
        check_strict_json_compat({
            'a': [1, 2, 3],
            'b': [3,],
            'c': [1, 2, 3, 4]
        }, Dict[str, List[int]]))
    self.assertTrue(
        check_strict_json_compat({
            'a': [1, 2, 3.],
            'b': [3,],
            'd': [1, 2, 3, 4]
        }, Dict[str, List[Union[int, float]]]))
    self.assertTrue(
        check_strict_json_compat({
            'a': {
                'b': True,
                'c': False
            },
            'b': None
        }, Dict[str, Optional[Dict[str, bool]]]))
    self.assertTrue(
        check_strict_json_compat([1, {
            'a': True
        }, None, True, [3., 4.]], List[Optional[Union[int, Dict[str, bool],
                                                      List[float], bool]]]))
    self.assertTrue(
        check_strict_json_compat({'a': [1, 2, 3]},
                                 Union[List[int], Dict[str, List[int]],
                                       Dict[str, float]]))
    self.assertTrue(
        check_strict_json_compat([1, 2, 3], Union[List[int], Dict[str,
                                                                  List[int]],
                                                  Dict[str, float]]))
    self.assertTrue(
        check_strict_json_compat({'a': 1.}, Union[List[int], Dict[str,
                                                                  List[int]],
                                                  Dict[str, float]]))
    self.assertTrue(check_strict_json_compat([1, 2, 3], List[Any]))
    self.assertTrue(check_strict_json_compat([], List[Any]))
    self.assertTrue(check_strict_json_compat({}, Dict[str, Any]))
    self.assertTrue(check_strict_json_compat(None, Optional[Dict[str, float]]))
    self.assertTrue(check_strict_json_compat([None, None], List[Any]))
    self.assertTrue(
        check_strict_json_compat([None, None], List[Optional[float]]))
    self.assertTrue(check_strict_json_compat({'a': 1., 'b': True}, Dict))
    self.assertTrue(
        check_strict_json_compat({
            'a': {
                'a': 3
            },
            'c': {
                'c': {
                    'c': 1
                }
            }
        }, Dict[str, Dict[str, Any]]))
    self.assertTrue(check_strict_json_compat(None, Optional[Dict[str, int]]))
    self.assertTrue(check_strict_json_compat([1, 2], Optional[List[int]]))
    self.assertTrue(check_strict_json_compat([None], List[Optional[int]]))
    self.assertTrue(
        check_strict_json_compat([[1, 2], [[3]], [4]],
                                 List[List[Union[List[int], int]]]))
    self.assertTrue(
        check_strict_json_compat({
            'a': 1.,
            'b': 2.
        }, Dict[str, float]))

    self.assertFalse(
        check_strict_json_compat({'a': [1, 2, 3.]}, Dict[str, List[int]]))
    self.assertFalse(
        check_strict_json_compat({
            'a': {
                'b': True,
                'c': False
            },
            'b': 1
        }, Dict[str, Optional[Dict[str, bool]]]))
    self.assertFalse(
        check_strict_json_compat({'a': [True, False]},
                                 Union[List[int], Dict[str, List[int]],
                                       Dict[str, float]]))
    self.assertFalse(
        check_strict_json_compat([b'123'], List[Union[int, float, bool, str]]))
    self.assertFalse(check_strict_json_compat({1: 2}, Dict[str, Any]))
    self.assertFalse(
        check_strict_json_compat([{
            'a': b'b'
        }], Dict[str, Union[int, float, bool, str]]))
    self.assertFalse(check_strict_json_compat(None, Dict[str, Any]))
    self.assertFalse(
        check_strict_json_compat([1, 2], List[Union[str, float, bool]]))
    self.assertFalse(
        check_strict_json_compat({
            'a': True,
            'b': False
        }, Dict[str, Union[int, float, str]]))
    self.assertFalse(
        check_strict_json_compat({
            'a': {
                'a': 3
            },
            'c': [1, 2]
        }, Dict[str, Dict[str, Any]]))
    self.assertFalse(check_strict_json_compat({'a': 1}, List[Any]))
    self.assertFalse(check_strict_json_compat([1.], Dict[str, Any]))
    self.assertFalse(
        check_strict_json_compat({
            'a': True,
            'b': 2.
        }, Dict[str, Union[int, float, str]]))


if __name__ == '__main__':
  tf.test.main()
