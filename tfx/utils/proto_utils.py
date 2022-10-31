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

from typing import Any, Dict, Iterator, TypeVar, Optional

from google.protobuf import any_pb2
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor as descriptor_lib
from google.protobuf import descriptor_pool
from google.protobuf import json_format
from google.protobuf import message
from google.protobuf import message_factory


def gather_file_descriptors(
    file_descriptor: descriptor_lib.FileDescriptor
) -> Iterator[descriptor_lib.FileDescriptor]:
  """Yields the file descriptor and all of its dependencies.

  Args:
    file_descriptor: The proto descriptor to start the dependency search from.

  Yields:
    All file descriptors in the transitive dependencies of the input descriptor,
    in topological order (i.e. dependencies before dependents).
    Each file descriptor is returned only once.
  """
  visited_files = set()  # To avoid duplicate outputs.

  def process_file(
      file: descriptor_lib.FileDescriptor
  ) -> Iterator[descriptor_lib.FileDescriptor]:
    """Yields the file's dependencies and then the file itself."""
    if file in visited_files:
      return
    visited_files.add(file)

    for dependency_file in file.dependencies:
      yield from process_file(dependency_file)
    yield file  # It's important that this is last.

  # Kick off the recursion.
  yield from process_file(file_descriptor)


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


def build_file_descriptor_set(
    pb_message: message.Message, fd_set: descriptor_pb2.FileDescriptorSet
) -> descriptor_pb2.FileDescriptorSet:
  """Builds file descriptor set for input pb message."""
  for fd in gather_file_descriptors(pb_message.DESCRIPTOR.file):
    fd.CopyToProto(fd_set.file.add())
  return fd_set


def _create_proto_instance_from_name(
    message_name: str, pool: descriptor_pool.DescriptorPool) -> ProtoMessage:
  """Creates a protobuf message instance from a given message name."""
  message_descriptor = pool.FindMessageTypeByName(message_name)
  factory = message_factory.MessageFactory(pool)
  message_type = factory.GetPrototype(message_descriptor)
  return message_type()


def deserialize_proto_message(
    serialized_message: str,
    message_name: str,
    file_descriptors: Optional[descriptor_pb2.FileDescriptorSet] = None
) -> ProtoMessage:
  """Converts serialized pb message string to its original message."""
  pool = descriptor_pool.Default()
  if file_descriptors:
    for file_descriptor in file_descriptors.file:
      pool.Add(file_descriptor)

  proto_instance = _create_proto_instance_from_name(message_name, pool)
  return json_format.Parse(
      serialized_message, proto_instance, descriptor_pool=pool)


def unpack_proto_any(any_proto: any_pb2.Any) -> ProtoMessage:
  """Unpacks a google.protobuf.Any message into its concrete type."""
  pool = descriptor_pool.Default()
  message_name = any_proto.type_url.split('/')[-1]
  proto_instance = _create_proto_instance_from_name(message_name, pool)
  any_proto.Unpack(proto_instance)
  return proto_instance
