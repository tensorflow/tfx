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
"""Deprecated location for TFX placeholders.

Please use the new module location at `tfx.dsl.components.placeholders`.
"""

from tfx.dsl.components import placeholders
from tfx.utils import deprecation_utils

ProtoSerializationFormat = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.ProtoSerializationFormat',
    name='tfx.dsl.components.placeholders.ProtoSerializationFormat',
    func_or_class=placeholders.ProtoSerializationFormat)

Placeholder = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.Placeholder',
    name='tfx.dsl.components.placeholders.Placeholder',
    func_or_class=placeholders.Placeholder)

ArtifactPlaceholder = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.ArtifactPlaceholder',
    name='tfx.dsl.components.placeholders.ArtifactPlaceholder',
    func_or_class=placeholders.ArtifactPlaceholder)

ExecPropertyPlaceholder = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.ExecPropertyPlaceholder',
    name='tfx.dsl.components.placeholders.ExecPropertyPlaceholder',
    func_or_class=placeholders.ExecPropertyPlaceholder)

RuntimeInfoPlaceholder = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.RuntimeInfoPlaceholder',
    name='tfx.dsl.components.placeholders.RuntimeInfoPlaceholder',
    func_or_class=placeholders.RuntimeInfoPlaceholder)

ExecInvocationPlaceholder = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.ExecInvocationPlaceholder',
    name='tfx.dsl.components.placeholders.ExecInvocationPlaceholder',
    func_or_class=placeholders.ExecInvocationPlaceholder)

input = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name,redefined-builtin
    deprecated_name='tfx.dsl.placeholder.placeholder.input',
    name='tfx.dsl.components.placeholders.input',
    func_or_class=placeholders.input)

output = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.output',
    name='tfx.dsl.components.placeholders.output',
    func_or_class=placeholders.output)

exec_property = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.exec_property',
    name='tfx.dsl.components.placeholders.exec_property',
    func_or_class=placeholders.exec_property)

RuntimeInfoKey = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.RuntimeInfoKey',
    name='tfx.dsl.components.placeholders.RuntimeInfoKey',
    func_or_class=placeholders.RuntimeInfoKey)

runtime_info = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.runtime_info',
    name='tfx.dsl.components.placeholders.runtime_info',
    func_or_class=placeholders.runtime_info)

execution_invocation = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.dsl.placeholder.placeholder.execution_invocation',
    name='tfx.dsl.components.placeholders.execution_invocation',
    func_or_class=placeholders.execution_invocation)
