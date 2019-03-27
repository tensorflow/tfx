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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys
import tempfile
import six
import tensorflow as tf
from typing import Callable
from typing import List
from typing import Text

from google.protobuf import text_format
from google.protobuf.message import Message
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


def import_func(module_path, fn_name):  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file."""

  # If a GCS bucket (gs://...), download to local filesystem first as
  # importlib can't import from GCS
  if module_path.startswith('gs://'):
    module_filename = os.path.basename(module_path)
    copy_file(module_path, module_filename, True)
    module_path = module_filename

  try:
    if six.PY2:
      import imp  # pylint: disable=g-import-not-at-top
      user_module = imp.load_source('user_module', module_path)
    else:
      import importlib.util  # pylint: disable=g-import-not-at-top
      spec = importlib.util.spec_from_file_location('user_module', module_path)
      user_module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(user_module)  # pytype: disable=attribute-error
  except IOError:
    raise IOError('{} not found in import_func()'.format(module_path))

  return getattr(user_module, fn_name)


def copy_file(src, dst, overwrite = False):
  """Copies a single file from source to destination."""

  if overwrite and tf.gfile.Exists(dst):
    tf.gfile.Remove(dst)
  dst_dir = os.path.dirname(dst)
  tf.gfile.MakeDirs(dst_dir)
  tf.gfile.Copy(src, dst, overwrite=overwrite)


def copy_dir(src, dst):
  """Copies the whole directory recursively from source to distination."""

  if tf.gfile.Exists(dst):
    tf.gfile.DeleteRecursively(dst)
  tf.gfile.MakeDirs(dst)

  for dir_name, sub_dirs, leaf_files in tf.gfile.Walk(src):
    for leaf_file in leaf_files:
      leaf_file_path = os.path.join(dir_name, leaf_file)
      new_file_path = os.path.join(dir_name.replace(src, dst, 1), leaf_file)
      tf.gfile.Copy(leaf_file_path, new_file_path)

    for sub_dir in sub_dirs:
      tf.gfile.MakeDirs(os.path.join(dst, sub_dir))


def get_only_uri_in_dir(dir_path):
  """Gets the only uri from given directory."""

  files = tf.gfile.ListDirectory(dir_path)
  if len(files) != 1:
    raise RuntimeError(
        'Only one file per dir is supported: {}.'.format(dir_path))
  filename = os.path.dirname(os.path.join(files[0], ''))
  return os.path.join(dir_path, filename)


def delete_dir(path):
  """Deletes a directory if exists."""

  if tf.gfile.IsDirectory(path):
    tf.gfile.DeleteRecursively(path)


def write_string_file(file_name, string_value):
  """Writes a string to file."""

  tf.gfile.MakeDirs(os.path.dirname(file_name))
  file_io.write_string_to_file(file_name, string_value)


def write_pbtxt_file(file_name, proto):
  """Writes a text proto to file."""

  write_string_file(file_name, text_format.MessageToString(proto))


def write_tfrecord_file(file_name, proto):
  """Writes a serialized tfrecord to file."""

  tf.gfile.MakeDirs(os.path.dirname(file_name))
  with tf.python_io.TFRecordWriter(file_name) as writer:
    writer.write(proto.SerializeToString())


def parse_pbtxt_file(file_name, message):
  """Parses a proto message from a text file and return message itself."""
  contents = file_io.read_file_to_string(file_name)
  text_format.Parse(contents, message)
  return message


def load_csv_column_names(csv_file):
  """Parse the first line of a csv file as column names."""
  with file_io.FileIO(csv_file, 'r') as f:
    return f.readline().strip().split(',')


def all_files_pattern(file_pattern):
  """Returns file pattern suitable for beam to locate multiple files."""
  return '{}*'.format(file_pattern)


class SchemaReader(object):
  """Schema reader."""

  def read(self, schema_path):
    """Gets a tf.metadata schema.

    Args:
      schema_path: Path to schema file.

    Returns:
      A tf.metadata schema.
    """

    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(schema_path)
    text_format.Parse(contents, result)
    return result


def build_package():
  """Builds a package using setuptools to be pushed to GCP."""
  # TODO(b/124821007): Revisit this once PyPi package exists
  src_dir = sys.modules['tfx'].__file__  # Get the install path for TFX
  setup_file = os.path.join(
      os.path.dirname(os.path.dirname(src_dir)), 'setup.py')
  # Create the temp package
  tmp_dir = tempfile.mkdtemp()
  cmd = ['python', setup_file, '--quiet', 'sdist', '--dist-dir', tmp_dir]
  subprocess.call(cmd)

  files = tf.gfile.ListDirectory(tmp_dir)
  if len(files) != 1:
    raise RuntimeError('Found multiple package files: {}'.format(tmp_dir))

  return os.path.join(tmp_dir, files[0])
