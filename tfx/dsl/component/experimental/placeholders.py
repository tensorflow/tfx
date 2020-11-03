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
"""Deprecated alias for `tfx.dsl.components.placeholders`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Standard Imports

from tfx.dsl.components import placeholders

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

InputValuePlaceholder = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.placeholders.InputValuePlaceholder',
    name='tfx.components.placeholders.InputValuePlaceholder',
    func_or_class=placeholders.InputValuePlaceholder)

InputUriPlaceholder = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.placeholders.InputUriPlaceholder',
    name='tfx.components.placeholders.InputUriPlaceholder',
    func_or_class=placeholders.InputUriPlaceholder)

OutputUriPlaceholder = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.placeholders.OutputUriPlaceholder',
    name='tfx.components.placeholders.OutputUriPlaceholder',
    func_or_class=placeholders.OutputUriPlaceholder)

ConcatPlaceholder = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.placeholders.ConcatPlaceholder',
    name='tfx.components.placeholders.ConcatPlaceholder',
    func_or_class=placeholders.ConcatPlaceholder)

CommandlineArgumentType = placeholders.CommandlineArgumentType
