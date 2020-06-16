# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Abstract TFX executor class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.components.base.executor_spec import ExecutorClassSpec
# from tfx.components.base.base_executor import DummyExecutor
#from unittest.mock import patch, MagicMock
# from unittest.mock import patch, Mock
import mock

class FakeComponentExecutorFactory(object):
  # @mock.patch('tfx.components.base.base_executor.EmptyExecutor')
  def __call__(self, executor_context):
    return mock.Mock()

class FakeExecutorClassSpec(ExecutorClassSpec):
  def __init__(self, fake_object_factory: FakeComponentExecutorFactory):
    self.executor_class = fake_object_factory
