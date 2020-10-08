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
"""Utility functions related to Examples artifact shared by components."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

from absl import logging
from tfx import types
from tfx.components.example_gen import utils as example_gen_utils
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts

_DEFAULT_PAYLOAD_FORMAT = example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE


def get_payload_format(examples: types.Artifact) -> int:
  """Returns the payload format of Examples artifact.

  If Examples artifact does not contain the "payload_format" custom property,
  it is made before tfx supports multiple payload format, and can regard as
  tf.Example format.

  Args:
    examples: A standard_artifacts.Examples artifact.

  Returns:
    payload_format: One of the enums in example_gen_pb2.PayloadFormat.
  """
  assert examples.type_name == standard_artifacts.Examples.TYPE_NAME, (
      'examples must be of type standard_artifacts.Examples')
  if examples.has_custom_property(
      example_gen_utils.PAYLOAD_FORMAT_PROPERTY_NAME):
    return example_gen_pb2.PayloadFormat.Value(
        examples.get_string_custom_property(
            example_gen_utils.PAYLOAD_FORMAT_PROPERTY_NAME))
  else:
    logging.warning('Examples artifact does not have %s custom property. '
                    'Falling back to %s',
                    example_gen_utils.PAYLOAD_FORMAT_PROPERTY_NAME,
                    example_gen_pb2.PayloadFormat.Name(_DEFAULT_PAYLOAD_FORMAT))
    return _DEFAULT_PAYLOAD_FORMAT


def get_payload_format_string(examples: types.Artifact) -> Text:
  """Returns the payload format as a string."""
  return example_gen_pb2.PayloadFormat.Name(get_payload_format(examples))


def set_payload_format(examples: types.Artifact, payload_format: int):
  """Sets the payload format custom property for `examples`.

  Args:
    examples: A standard_artifacts.Examples artifact.
    payload_format: One of the enums in example_gen_pb2.PayloadFormat.
  """
  assert examples.type_name == standard_artifacts.Examples.TYPE_NAME, (
      'examples must be of type standard_artifacts.Examples')
  examples.set_string_custom_property(
      example_gen_utils.PAYLOAD_FORMAT_PROPERTY_NAME,
      example_gen_pb2.PayloadFormat.Name(payload_format))
