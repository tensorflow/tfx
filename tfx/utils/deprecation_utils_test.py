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
"""Tests for tfx.utils.deprecation_utils."""

from unittest import mock

import tensorflow as tf
from tfx.utils import deprecation_utils
from tfx.utils import test_case_utils


class DependencyUtilsTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    deprecation_utils._PRINTED_WARNING = set()
    self._mock_warn = self.enter_context(mock.patch('warnings.warn'))

  def _assertDeprecatedWarningRegex(self, expected_regex):
    self._mock_warn.assert_called()
    (message, warning_cls), unused_kwargs = self._mock_warn.call_args
    self.assertEqual(warning_cls, deprecation_utils.TfxDeprecationWarning)
    self.assertRegex(message, expected_regex)

  def _mock_function(self, name='function'):
    """Return a mock function."""
    function = mock.MagicMock()
    # Either `__qualname__` or `__name__` is expected to be set for a function.
    setattr(function, '__qualname__', name)
    setattr(function, '__name__', name)
    return function

  def testDeprecated(self):
    # By default, we warn once across all calls.
    my_function_1 = self._mock_function(name='my_function_1')
    deprecated_func_1 = deprecation_utils.deprecated(
        '2099-01-02', 'Please change to new_my_function_1')(
            my_function_1)
    deprecated_func_1()
    deprecated_func_1()
    self._assertDeprecatedWarningRegex(
        r'From .*: my_function_1 \(from .*\) is deprecated and will be '
        r'removed after 2099-01-02. Instructions for updating:\n'
        r'Please change to new_my_function_1')
    self.assertEqual(my_function_1.call_count, 2)
    self._mock_warn.reset_mock()

    # If `warn_once=False`, we warn once for each call.
    my_function_2 = self._mock_function()
    deprecated_func_2 = deprecation_utils.deprecated(
        '2099-01-02', 'Please change to new_my_function_2', warn_once=False)(
            my_function_2)
    deprecated_func_2()
    deprecated_func_2()
    deprecated_func_2()
    self.assertEqual(self._mock_warn.call_count, 3)
    self.assertEqual(my_function_2.call_count, 3)

  def testDeprecationAliasFunction(self):
    # By default, we warn once across all calls.
    my_function_1 = self._mock_function(name='my_function_1')
    deprecation_alias_1 = deprecation_utils.deprecated_alias(
        'deprecation_alias_1', 'my_function_1', my_function_1)
    deprecation_alias_1()
    deprecation_alias_1()
    self._assertDeprecatedWarningRegex(
        'From .*: The name deprecation_alias_1 is deprecated. Please use '
        'my_function_1 instead.')
    self.assertEqual(my_function_1.call_count, 2)
    self._mock_warn.reset_mock()

    # If `warn_once=False`, we warn once for each call.
    my_function_2 = self._mock_function()
    deprecation_alias_2 = deprecation_utils.deprecated_alias(
        'deprecation_alias_2', 'my_function_2', my_function_2, warn_once=False)
    deprecation_alias_2()
    deprecation_alias_2()
    deprecation_alias_2()
    self.assertEqual(self._mock_warn.call_count, 3)
    self.assertEqual(my_function_2.call_count, 3)

  def testDeprecationClass(self):

    class MyClass1:
      __init__ = mock.MagicMock()

    class MyClass2:
      __init__ = mock.MagicMock()

    # By default, we warn once across all calls.
    DeprecatedAliasClass1 = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
        'DeprecatedAliasClass1', 'MyClass1', MyClass1)
    DeprecatedAliasClass1()
    DeprecatedAliasClass1()
    self._assertDeprecatedWarningRegex(
        'From .*: The name DeprecatedAliasClass1 is deprecated. Please use '
        'MyClass1 instead.')
    self.assertEqual(MyClass1.__init__.call_count, 2)
    self._mock_warn.reset_mock()

    # Check properties of the deprecated class.
    self.assertEqual(DeprecatedAliasClass1.__name__, '_NewDeprecatedClass')
    self.assertEqual(
        repr(DeprecatedAliasClass1.__doc__),
        repr("""DEPRECATED CLASS

Warning: THIS CLASS IS DEPRECATED. It will be removed in a future version.
Please use MyClass1 instead."""))

    # If `warn_once=False`, we warn once for each call.
    DeprecatedAliasClass2 = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
        'DeprecatedAliasClass2',
        'MyClass2',
        MyClass2,
        warn_once=False)
    DeprecatedAliasClass2()
    DeprecatedAliasClass2()
    DeprecatedAliasClass2()
    self.assertEqual(self._mock_warn.call_count, 3)
    self.assertEqual(MyClass2.__init__.call_count, 3)


if __name__ == '__main__':
  tf.test.main()
