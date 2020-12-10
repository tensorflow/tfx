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
"""Tensorflow GFile-based filesystem plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Iterable, List, Text, Tuple
import tensorflow as tf

from tfx.dsl.io import filesystem
from tfx.dsl.io import filesystem_registry
from tfx.dsl.io.filesystem import PathType


class TensorflowFilesystem(filesystem.Filesystem):
  """Filesystem that delegates to `tensorflow.io.gfile`."""

  SUPPORTED_SCHEMES = ['', 'gs://', 'hdfs://', 's3://']

  @staticmethod
  def open(name: PathType, mode: Text = 'r') -> Any:
    return tf.io.gfile.GFile(name, mode=mode)

  @staticmethod
  def copy(src: PathType, dst: PathType, overwrite: bool = False) -> None:
    tf.io.gfile.copy(src, dst, overwrite=overwrite)

  @staticmethod
  def exists(path: PathType) -> bool:
    return tf.io.gfile.exists(path)

  @staticmethod
  def glob(pattern: PathType) -> List[PathType]:
    return tf.io.gfile.glob(pattern)

  @staticmethod
  def isdir(path: PathType) -> bool:
    return tf.io.gfile.isdir(path)

  @staticmethod
  def listdir(path: PathType) -> List[PathType]:
    return tf.io.gfile.listdir(path)

  @staticmethod
  def makedirs(path: PathType) -> None:
    tf.io.gfile.makedirs(path)

  @staticmethod
  def mkdir(path: PathType) -> None:
    tf.io.gfile.mkdir(path)

  @staticmethod
  def remove(path: PathType) -> None:
    tf.io.gfile.remove(path)

  @staticmethod
  def rename(src: PathType, dst: PathType, overwrite: bool = False) -> None:
    tf.io.gfile.rename(src, dst, overwrite=overwrite)

  @staticmethod
  def rmtree(path: PathType) -> None:
    tf.io.gfile.rmtree(path)

  @staticmethod
  def stat(path: PathType) -> Any:
    return tf.io.gfile.stat(path)

  @staticmethod
  def walk(
      top: PathType,
      topdown: bool = True,
      onerror: Callable[..., None] = None
  ) -> Iterable[Tuple[PathType, List[PathType], List[PathType]]]:
    yield from tf.io.gfile.walk(top, topdown=topdown, onerror=onerror)


filesystem_registry.DEFAULT_FILESYSTEM_REGISTRY.register(
    TensorflowFilesystem, priority=0, use_as_fallback=True)
