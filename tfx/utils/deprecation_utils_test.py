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

# Standard Imports

import mock
import tensorflow as tf

from tfx.utils import deprecation_utils


class DependencyUtilsTest(tf.test.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    deprecation_utils._PRINTED_WARNING = {}

  def _mock_function(self):
    """Return a mock function."""
    function = mock.MagicMock()
    # Either `__qualname__` or `__name__` is expected to be set for a function.
    setattr(function, '__qualname__', 'function')
    return function

  @mock.patch('absl.logging.warning')
  def testDeprecated(self, mock_absl_warning):
    # By default, we warn once across all calls.
    my_function_1 = self._mock_function()
    deprecated_func_1 = deprecation_utils.deprecated(
        '2099-01-02', 'Please change to new_my_function_1')(
            my_function_1)
    deprecated_func_1()
    deprecated_func_1()
    mock_absl_warning.assert_called_once_with(
        mock.ANY, mock.ANY, mock.ANY, mock.ANY, 'after 2099-01-02',
        'Please change to new_my_function_1')
    self.assertEqual(my_function_1.call_count, 2)
    mock_absl_warning.reset_mock()

    # If `warn_once=False`, we warn once for each call.
    my_function_2 = self._mock_function()
    deprecated_func_2 = deprecation_utils.deprecated(
        '2099-01-02', 'Please change to new_my_function_2', warn_once=False)(
            my_function_2)
    deprecated_func_2()
    deprecated_func_2()
    deprecated_func_2()
    self.assertEqual(mock_absl_warning.call_count, 3)
    self.assertEqual(my_function_2.call_count, 3)

  @mock.patch('absl.logging.warning')
  def testDeprecationAliasFunction(self, mock_absl_warning):
    # By default, we warn once across all calls.
    my_function_1 = self._mock_function()
    deprecation_alias_1 = deprecation_utils.deprecated_alias(
        'deprecation_alias_1', 'my_function_1', my_function_1)
    deprecation_alias_1()
    deprecation_alias_1()
    mock_absl_warning.assert_called_once_with(mock.ANY, mock.ANY,
                                              'deprecation_alias_1',
                                              'my_function_1')
    self.assertEqual(my_function_1.call_count, 2)
    mock_absl_warning.reset_mock()

    # If `warn_once=False`, we warn once for each call.
    my_function_2 = self._mock_function()
    deprecation_alias_2 = deprecation_utils.deprecated_alias(
        'deprecation_alias_2', 'my_function_2', my_function_2, warn_once=False)
    deprecation_alias_2()
    deprecation_alias_2()
    deprecation_alias_2()
    self.assertEqual(mock_absl_warning.call_count, 3)
    self.assertEqual(my_function_2.call_count, 3)
    mock_absl_warning.reset_mock()

  @mock.patch('absl.logging.warning')
  def testDeprecationClass(self, mock_absl_warning):

    class MyClass1(object):
      __init__ = mock.MagicMock()

    class MyClass2(object):
      __init__ = mock.MagicMock()

    # By default, we warn once across all calls.
    DeprecatedAliasClass1 = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
        'DeprecatedAliasClass1', 'MyClass1', MyClass1)
    DeprecatedAliasClass1()
    DeprecatedAliasClass1()
    mock_absl_warning.assert_called_once_with(mock.ANY, mock.ANY,
                                              'DeprecatedAliasClass1',
                                              'MyClass1')
    self.assertEqual(MyClass1.__init__.call_count, 2)
    mock_absl_warning.reset_mock()

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
    self.assertEqual(mock_absl_warning.call_count, 3)
    self.assertEqual(MyClass2.__init__.call_count, 3)
    mock_absl_warning.reset_mock()


if __name__ == '__main__':
  tf.test.main()
