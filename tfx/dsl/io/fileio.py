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
"""Pluggable file I/O interface for use in TFX system and components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Iterable, List, Text, Tuple, Type

from tfx.dsl.io import filesystem
from tfx.dsl.io import filesystem_registry
from tfx.dsl.io.filesystem import PathType

# Import modules that may provide filesystem plugins.
import tfx.dsl.io.plugins.tensorflow_gfile  # pylint: disable=unused-import, g-import-not-at-top
import tfx.dsl.io.plugins.local  # pylint: disable=unused-import, g-import-not-at-top


# Expose `NotFoundError` as `fileio.NotFoundError`.
NotFoundError = filesystem.NotFoundError


def _get_filesystem(path) -> Type[filesystem.Filesystem]:
  return (filesystem_registry.DEFAULT_FILESYSTEM_REGISTRY
          .get_filesystem_for_path(path))


def open(path: PathType, mode: Text = 'r'):  # pylint: disable=redefined-builtin
  """Open a file at the given path."""
  return _get_filesystem(path).open(path, mode=mode)


def copy(src: PathType, dst: PathType, overwrite: bool = False) -> None:
  """Copy a file from the source to the destination."""
  src_fs = _get_filesystem(src)
  dst_fs = _get_filesystem(dst)
  if src_fs is dst_fs:
    src_fs.copy(src, dst, overwrite=overwrite)
  else:
    if not overwrite and exists(dst):
      raise OSError(
          ('Destination file %r already exists and argument `overwrite` is '
           'false.') % dst)
    contents = open(src, mode='rb').read()
    open(dst, mode='wb').write(contents)


def exists(path: PathType) -> bool:
  """Return whether a path exists."""
  return _get_filesystem(path).exists(path)


def glob(pattern: PathType) -> List[PathType]:
  """Return the paths that match a glob pattern."""
  return _get_filesystem(pattern).glob(pattern)


def isdir(path: PathType) -> bool:
  """Return whether a path is a directory."""
  return _get_filesystem(path).isdir(path)


def listdir(path: PathType) -> List[PathType]:
  """Return the list of files in a directory."""
  return _get_filesystem(path).listdir(path)


def makedirs(path: PathType) -> None:
  """Make a directory at the given path, recursively creating parents."""
  _get_filesystem(path).makedirs(path)


def mkdir(path: PathType) -> None:
  """Make a directory at the given path; parent directory must exist."""
  _get_filesystem(path).mkdir(path)


def remove(path: PathType) -> None:
  """Remove the file at the given path."""
  _get_filesystem(path).remove(path)


def rename(src: PathType, dst: PathType, overwrite: bool = False) -> None:
  """Rename a source file to a destination path."""
  src_fs = _get_filesystem(src)
  dst_fs = _get_filesystem(dst)
  if src_fs is dst_fs:
    src_fs.rename(src, dst, overwrite=overwrite)
  else:
    raise NotImplementedError(
        ('Rename from %r to %r using different filesystems plugins is '
         'currently not supported.') % (src, dst))


def rmtree(path: PathType) -> None:
  """Remove the given directory and its recursive contents."""
  _get_filesystem(path).rmtree(path)


def stat(path: PathType) -> Any:
  """Return the stat descriptor for a given file path."""
  return _get_filesystem(path).stat(path)


def walk(
    top: PathType,
    topdown: bool = True,
    onerror: Callable[..., None] = None
) -> Iterable[Tuple[PathType, List[PathType], List[PathType]]]:
  """Return an iterator walking a directory tree."""
  return _get_filesystem(top).walk(top, topdown=topdown, onerror=onerror)
