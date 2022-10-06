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

from typing import Any, Callable, Iterable, List, Optional, Tuple

from tfx.dsl.io import filesystem
from tfx.dsl.io import filesystem_registry
from tfx.dsl.io.filesystem import PathType

try:
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  tf = None

try:
  import tensorflow_io as _  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  pass

if tf:

  class TensorflowFilesystem(filesystem.Filesystem):
    """Filesystem that delegates to `tensorflow.io.gfile`."""

    SUPPORTED_SCHEMES = ['', 'gs://', 'hdfs://', 's3://']

    @staticmethod
    def open(name: PathType, mode: str = 'r') -> Any:
      # Because the GFile implementation delays I/O until necessary, we cannot
      # catch `NotFoundError` here.
      return tf.io.gfile.GFile(name, mode=mode)

    @staticmethod
    def copy(src: PathType, dst: PathType, overwrite: bool = False) -> None:
      try:
        tf.io.gfile.copy(src, dst, overwrite=overwrite)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def exists(path: PathType) -> bool:
      return tf.io.gfile.exists(path)

    @staticmethod
    def glob(pattern: PathType) -> List[PathType]:
      try:
        return tf.io.gfile.glob(pattern)
      except tf.errors.NotFoundError:
        return []

    @staticmethod
    def isdir(path: PathType) -> bool:
      return tf.io.gfile.isdir(path)

    @staticmethod
    def listdir(path: PathType) -> List[PathType]:
      try:
        return tf.io.gfile.listdir(path)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def makedirs(path: PathType) -> None:
      tf.io.gfile.makedirs(path)

    @staticmethod
    def mkdir(path: PathType) -> None:
      try:
        tf.io.gfile.mkdir(path)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def remove(path: PathType) -> None:
      try:
        tf.io.gfile.remove(path)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def rename(src: PathType, dst: PathType, overwrite: bool = False) -> None:
      try:
        tf.io.gfile.rename(src, dst, overwrite=overwrite)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def rmtree(path: PathType) -> None:
      try:
        tf.io.gfile.rmtree(path)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def stat(path: PathType) -> Any:
      try:
        return tf.io.gfile.stat(path)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

    @staticmethod
    def walk(
        top: PathType,
        topdown: bool = True,
        onerror: Optional[Callable[..., None]] = None
    ) -> Iterable[Tuple[PathType, List[PathType], List[PathType]]]:
      try:
        yield from tf.io.gfile.walk(top, topdown=topdown, onerror=onerror)
      except tf.errors.NotFoundError as e:
        raise filesystem.NotFoundError() from e

  filesystem_registry.DEFAULT_FILESYSTEM_REGISTRY.register(
      TensorflowFilesystem, priority=10, use_as_fallback=True)
else:
  TensorflowFilesystem = None  # pylint: disable=invalid-name
