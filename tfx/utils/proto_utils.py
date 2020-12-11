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
"""Utilities for proto related manipulations."""

import itertools
from typing import Any, Dict, Iterator, TypeVar

from google.protobuf import descriptor as descriptor_lib
from google.protobuf import json_format
from google.protobuf import message


def gather_file_descriptors(
    descriptor: descriptor_lib.Descriptor,
    enable_extensions: bool = False) -> Iterator[descriptor_lib.FileDescriptor]:
  """Yield all depdendent file descriptors of a given proto descriptor.

  Args:
    descriptor: The proto descriptor to start the dependency search from.
    enable_extensions: Optional. True if proto extensions are enabled. Default
      to False.

  Yields:
    All file descriptors in the transitive dependencies of descriptor.
    Each file descriptor is returned only once.
  """
  visited_files = set()
  visited_messages = set()
  messages = [descriptor]

  # Walk in depth through all the fields and extensions of the given descriptor
  # and all the referenced messages.
  while messages:
    descriptor = messages.pop()
    visited_files.add(descriptor.file)

    if enable_extensions:
      extensions = descriptor.file.pool.FindAllExtensions(descriptor)
    else:
      extensions = []
    for field in itertools.chain(descriptor.fields, extensions):
      if field.message_type and field.message_type not in visited_messages:
        visited_messages.add(field.message_type)
        messages.append(field.message_type)

    for extension in extensions:
      # Note: extension.file may differ from descriptor.file.
      visited_files.add(extension.file)

  # Go through the collected files and add their explicit dependencies.
  files = list(visited_files)
  while files:
    file_descriptor = files.pop()
    yield file_descriptor
    for dependency in file_descriptor.dependencies:
      if dependency not in visited_files:
        visited_files.add(dependency)
        files.append(dependency)


def proto_to_json(proto: message.Message) -> str:
  """Simple JSON Formatter wrapper for consistent formatting."""
  return json_format.MessageToJson(
      message=proto, sort_keys=True, preserving_proto_field_name=True)


def proto_to_dict(proto: message.Message) -> Dict[str, Any]:
  """Simple JSON Formatter wrapper for consistent formatting."""
  return json_format.MessageToDict(
      message=proto, preserving_proto_field_name=True)


# Type for a subclass of message.Message which will be used as a return type.
ProtoMessage = TypeVar('ProtoMessage', bound=message.Message)


def json_to_proto(json_str: str, proto: ProtoMessage) -> ProtoMessage:
  """Simple JSON Parser wrapper for consistent parsing."""
  return json_format.Parse(json_str, proto, ignore_unknown_fields=True)


def dict_to_proto(json_dict: Dict[Any, Any],
                  proto: ProtoMessage) -> ProtoMessage:
  """Simple JSON Parser wrapper for consistent parsing."""
  return json_format.ParseDict(json_dict, proto, ignore_unknown_fields=True)
