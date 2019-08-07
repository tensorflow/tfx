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

from typing import Text
from tfx import types
from tfx.types import channel_utils
from tfx.types import standard_artifacts


def external_input(uri: Text) -> types.Channel:
  """Helper function to declare external input.

  Args:
    uri: external path.

  Returns:
    input channel.
  """
  instance = standard_artifacts.ExternalArtifact()
  instance.uri = uri
  return channel_utils.as_channel([instance])


# TODO(b/136598840): deprecate this, use external_input.
def csv_input(uri: Text) -> types.Channel:
  """Helper function to declare input for csv_example_gen component.

  Args:
    uri: path of an external directory with a single csv file inside.

  Returns:
    input channel.
  """
  return external_input(uri)


# TODO(b/136598840): deprecate this, use external_input.
def tfrecord_input(uri: Text) -> types.Channel:
  """Helper function to declare input for import_example_gen component.

  Args:
    uri: path of an external directory with tfrecord(contains tf examples) files
      inside.

  Returns:
    input channel.
  """
  return external_input(uri)
