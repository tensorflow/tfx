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
"""Utilities for DSL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx import types
from tfx.types import channel_utils
from tfx.types import standard_artifacts


# TODO(b/158333888): deprecate external_input function.
@deprecation.deprecated(
    None, 'external_input is deprecated, directly pass the uri to ExampleGen.')
def external_input(uri: Any) -> types.Channel:
  """Helper function to declare external input.

  Args:
    uri: external path, can be RuntimeParameter

  Returns:
    input channel.
  """
  instance = standard_artifacts.ExternalArtifact()
  instance.uri = str(uri)
  return channel_utils.as_channel([instance])
