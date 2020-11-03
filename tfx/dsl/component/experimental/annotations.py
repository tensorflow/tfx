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
"""Deprecated alias for `tfx.dsl.components.annotations`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

from tfx.dsl.components import annotations

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

InputArtifact = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.annotations.InputArtifact',
    name='tfx.components.annotations.InputArtifact',
    func_or_class=annotations.InputArtifact)

OutputArtifact = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.annotations.OutputArtifact',
    name='tfx.components.annotations.OutputArtifact',
    func_or_class=annotations.OutputArtifact)

Parameter = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.annotations.Parameter',
    name='tfx.components.annotations.Parameter',
    func_or_class=annotations.Parameter)

OutputDict = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.component.experimental.annotations.OutputDict',
    name='tfx.components.annotations.OutputDict',
    func_or_class=annotations.OutputDict)
