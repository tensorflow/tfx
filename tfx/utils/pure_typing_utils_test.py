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
"""Tests for tfx.utils.pure_typing_utils."""

from typing import Optional, Union
import tensorflow as tf
from tfx.utils import pure_typing_utils


class PureTypingUtilsTest(tf.test.TestCase):

  def test_maybe_unwrap_optional(self):
    def assert_unwrapped(query, expected):
      unwrapped, result = pure_typing_utils.maybe_unwrap_optional(query)
      with self.subTest(f'unwrap_optional({query}) == {expected}'):
        self.assertTrue(unwrapped)
        self.assertEqual(result, expected)

    def assert_not_unwrapped(query):
      unwrapped, _ = pure_typing_utils.maybe_unwrap_optional(query)
      with self.subTest(f'{query} is not optional.'):
        self.assertFalse(unwrapped)

    assert_unwrapped(Optional[int], int)
    assert_unwrapped(Optional[list[int]], list[int])
    assert_unwrapped(Optional[list[Optional[int]]], list[Optional[int]])
    assert_unwrapped(Optional[Optional[int]], int)
    assert_unwrapped(Union[str, None], str)
    assert_unwrapped(Union[None, str], str)
    assert_unwrapped(Union[str, None, None], str)
    assert_not_unwrapped(str)
    assert_not_unwrapped(None)
    assert_not_unwrapped(Union[None, None])
    assert_not_unwrapped(Union[list, dict, None])
