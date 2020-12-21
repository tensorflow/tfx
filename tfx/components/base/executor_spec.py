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
"""Stub for pre-rename `tfx.dsl.components.base.executor_spec`."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.dsl.components.base import executor_spec
from tfx.utils import deprecation_utils

ExecutorSpec = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.components.base.executor_spec.ExecutorSpec',
    name='tfx.dsl.components.base.executor_spec.ExecutorSpec',
    func_or_class=executor_spec.ExecutorSpec)

ExecutorClassSpec = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.components.base.executor_spec.ExecutorClassSpec',
    name='tfx.dsl.components.base.executor_spec.ExecutorClassSpec',
    func_or_class=executor_spec.ExecutorClassSpec)

ExecutorContainerSpec = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.components.base.executor_spec.ExecutorContainerSpec',
    name='tfx.dsl.components.base.executor_spec.ExecutorContainerSpec',
    func_or_class=executor_spec.ExecutorContainerSpec)
