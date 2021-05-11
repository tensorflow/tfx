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
"""Utility class for I/O."""

import os
import tempfile
from typing import List, Text, TypeVar

from tfx.dsl.io import fileio
from google.protobuf import json_format
from google.protobuf import text_format
from google.protobuf.message import Message

try:
  from tensorflow_metadata.proto.v0.schema_pb2 import Schema as schema_pb2_Schema  # pylint: disable=g-import-not-at-top,g-importing-member
except ModuleNotFoundError as e:
  schema_pb2_Schema = None  # pylint: disable=invalid-name

# Nano seconds per second.
NANO_PER_SEC = 1000 * 1000 * 1000

# If path starts with one of those, consider files are in remote filesystem.
_REMOTE_FS_PREFIX = ['gs://', 'hdfs://', 's3://']


def ensure_local(file_path: Text) -> Text:
  """Ensures that the given file path is made available locally."""
  if not any([file_path.startswith(prefix) for prefix in _REMOTE_FS_PREFIX]):
    return file_path

  temp_dir = tempfile.mkdtemp()
  local_path = os.path.join(temp_dir, os.path.basename(file_path))
  copy_file(file_path, local_path, True)
  return local_path


def copy_file(src: Text, dst: Text, overwrite: bool = False):
  """Copies a single file from source to destination."""

  if overwrite and fileio.exists(dst):
    fileio.remove(dst)
  dst_dir = os.path.dirname(dst)
  fileio.makedirs(dst_dir)
  fileio.copy(src, dst, overwrite=overwrite)


def copy_dir(src: Text, dst: Text) -> None:
  """Copies the whole directory recursively from source to destination."""
  src = src.rstrip('/')
  dst = dst.rstrip('/')

  if fileio.exists(dst):
    fileio.rmtree(dst)
  fileio.makedirs(dst)

  for dir_name, sub_dirs, leaf_files in fileio.walk(src):
    for leaf_file in leaf_files:
      leaf_file_path = os.path.join(dir_name, leaf_file)
      new_file_path = os.path.join(dir_name.replace(src, dst, 1), leaf_file)
      fileio.copy(leaf_file_path, new_file_path)

    for sub_dir in sub_dirs:
      fileio.makedirs(os.path.join(dir_name.replace(src, dst, 1), sub_dir))


def get_only_uri_in_dir(dir_path: Text) -> Text:
  """Gets the only uri from given directory."""

  files = fileio.listdir(dir_path)
  if len(files) != 1:
    raise RuntimeError(
        'Only one file per dir is supported: {}.'.format(dir_path))
  filename = os.path.dirname(os.path.join(files[0], ''))
  return os.path.join(dir_path, filename)


def delete_dir(path: Text) -> None:
  """Deletes a directory if exists."""

  if fileio.isdir(path):
    fileio.rmtree(path)


def write_string_file(file_name: Text, string_value: Text) -> None:
  """Writes a string to file."""

  fileio.makedirs(os.path.dirname(file_name))
  with fileio.open(file_name, 'w') as f:
    f.write(string_value)


def write_bytes_file(file_name: Text, content: bytes) -> None:
  """Writes bytes to file."""

  fileio.makedirs(os.path.dirname(file_name))
  with fileio.open(file_name, 'wb') as f:
    f.write(content)


def write_pbtxt_file(file_name: Text, proto: Message) -> None:
  """Writes a text protobuf to file."""

  write_string_file(file_name, text_format.MessageToString(proto))


def write_tfrecord_file(file_name: Text, *proto: Message) -> None:
  """Writes a serialized tfrecord to file."""
  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top
  except ModuleNotFoundError as e:
    raise Exception(
        'TensorFlow must be installed to use this functionality.') from e
  fileio.makedirs(os.path.dirname(file_name))
  with tf.io.TFRecordWriter(file_name) as writer:
    for message in proto:
      writer.write(message.SerializeToString())


# Type for a subclass of message.Message which will be used as a return type.
ProtoMessage = TypeVar('ProtoMessage', bound=Message)


def parse_pbtxt_file(file_name: Text, message: ProtoMessage) -> ProtoMessage:
  """Parses a protobuf message from a text file and return message itself."""
  contents = fileio.open(file_name).read()
  text_format.Parse(contents, message)
  return message


def parse_json_file(file_name: Text, message: ProtoMessage) -> ProtoMessage:
  """Parses a protobuf message from a JSON file and return itself."""
  contents = fileio.open(file_name).read()
  json_format.Parse(contents, message)
  return message


def load_csv_column_names(csv_file: Text) -> List[Text]:
  """Parse the first line of a csv file as column names."""
  with fileio.open(csv_file) as f:
    return f.readline().strip().split(',')


def all_files_pattern(file_pattern: Text) -> Text:
  """Returns file pattern suitable for Beam to locate multiple files."""
  return os.path.join(file_pattern, '*')


def generate_fingerprint(split_name: Text, file_pattern: Text) -> Text:
  """Generates a fingerprint for all files that match the pattern."""
  files = fileio.glob(file_pattern)
  total_bytes = 0
  # Checksum used here is based on timestamp (mtime).
  # Checksums are xor'ed and sum'ed over the files so that they are order-
  # independent.
  xor_checksum = 0
  sum_checksum = 0
  for f in files:
    stat = fileio.stat(f)
    total_bytes += stat.length
    # Take mtime only up to second-granularity.
    mtime = int(stat.mtime_nsec / NANO_PER_SEC)
    xor_checksum ^= mtime
    sum_checksum += mtime

  return 'split:%s,num_files:%d,total_bytes:%d,xor_checksum:%d,sum_checksum:%d' % (
      split_name, len(files), total_bytes, xor_checksum, sum_checksum)


def read_string_file(file_name: Text) -> Text:
  """Reads a string from a file."""
  if not fileio.exists(file_name):
    msg = '{} does not exist'.format(file_name)
    raise FileNotFoundError(msg)
  return fileio.open(file_name).read()


def read_bytes_file(file_name: Text) -> bytes:
  """Reads bytes from a file."""
  if not fileio.exists(file_name):
    msg = '{} does not exist'.format(file_name)
    raise FileNotFoundError(msg)
  return fileio.open(file_name, 'rb').read()


class SchemaReader(object):
  """Schema reader."""

  def read(self, schema_path: Text) -> schema_pb2_Schema:  # pytype: disable=invalid-annotation
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """
    try:
      from tensorflow_metadata.proto.v0 import schema_pb2  # pylint: disable=g-import-not-at-top
    except ModuleNotFoundError as e:
      raise Exception('The full "tfx" package must be installed to use this '
                      'functionality.') from e

    result = schema_pb2.Schema()
    contents = fileio.open(schema_path).read()
    text_format.Parse(contents, result)
    return result
