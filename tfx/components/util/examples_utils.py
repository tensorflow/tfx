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

import json
import os
from typing import Dict, List, Optional, Tuple

from absl import logging
from tfx import types
from tfx.components.example_gen import utils as example_gen_utils
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils


_DEFAULT_PAYLOAD_FORMAT = example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE
_DEFAULT_FILE_FORMAT = 'tfrecords_gzip'

# If set, encodes a dictionary mapping split name to file pattern. This will
# be used by supported components instead of the pattern uri/Split-name/*.
# TODO(b/266823949): Support all first-party components.
CUSTOM_SPLIT_PATTERN_PROPERTY_NAME = 'custom_split_pattern'


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


def get_payload_format_string(examples: types.Artifact) -> str:
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


def get_file_format(examples: types.Artifact) -> str:
  """Returns the file format of Examples artifact.

  If Examples artifact does not contain the "file_format" custom property,
  it is made by OSS ExampleGen and can be treated as 'tfrecords_gzip' format.

  Args:
    examples: A standard_artifacts.Examples artifact.

  Returns:
    One of the file format that tfx_bsl understands.
  """
  assert examples.type_name == standard_artifacts.Examples.TYPE_NAME, (
      'examples must be of type standard_artifacts.Examples')
  if examples.has_custom_property(example_gen_utils.FILE_FORMAT_PROPERTY_NAME):
    return examples.get_string_custom_property(
        example_gen_utils.FILE_FORMAT_PROPERTY_NAME)
  else:
    return _DEFAULT_FILE_FORMAT


def set_file_format(examples: types.Artifact, file_format: str):
  """Sets the file format custom property for `examples`.

  Args:
    examples: A standard_artifacts.Examples artifact.
    file_format: One of the file format that tfx_bsl understands.
  """
  assert examples.type_name == standard_artifacts.Examples.TYPE_NAME, (
      'examples must be of type standard_artifacts.Examples')
  examples.set_string_custom_property(
      example_gen_utils.FILE_FORMAT_PROPERTY_NAME, file_format)


def get_custom_split_patterns_key_and_property(
    split_to_pattern: Dict[str, str]
) -> Tuple[str, str]:
  """Get a custom property name and value encoding custom split patterns.

  Args:
    split_to_pattern: A dictionary mapping split names to file patterns. These
      patterns should be relative to the artifact's uri, which is expected to be
      an ancestor directory of split patterns.

  Returns:
    A tuple consisting of a property name and value appropriate for
    artifact.set_string_custom_property.
  """
  return CUSTOM_SPLIT_PATTERN_PROPERTY_NAME, json.dumps(split_to_pattern)


def _get_custom_split_pattern(
    examples: types.Artifact, split: str
) -> Optional[str]:
  """Get a custom split file pattern for a split if set, or None."""
  if not examples.has_custom_property(CUSTOM_SPLIT_PATTERN_PROPERTY_NAME):
    return None
  custom_split_patterns = json.loads(
      examples.get_string_custom_property(CUSTOM_SPLIT_PATTERN_PROPERTY_NAME)
  )
  if split not in custom_split_patterns:
    raise ValueError(
        'Missing split %s for artifact with custom split pattern properties'
        % split
    )
  return os.path.join(examples.uri, custom_split_patterns[split])


def get_split_file_patterns(
    artifacts: List[types.Artifact], split: str
) -> List[str]:
  """Returns a file pattern for reading examples from a split."""
  result = []
  for artifact in artifacts:
    maybe_pattern = _get_custom_split_pattern(artifact, split)
    if maybe_pattern is not None:
      result.append(maybe_pattern)
    else:
      uri = artifact_utils.get_split_uri([artifact], split)
      result.append(io_utils.all_files_pattern(uri))
  return result
