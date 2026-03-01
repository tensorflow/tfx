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
"""Filesystem registry managing filesystem plugins."""

import re
import threading
from typing import Type

from tfx.dsl.io import filesystem
from tfx.dsl.io.filesystem import PathType


class FilesystemRegistry:
  """Registry of pluggable filesystem implementations used in TFX components."""

  def __init__(self):
    self._preferred_filesystem_by_scheme = {}
    self._filesystem_priority = {}
    self._fallback_filesystem = None
    self._registration_lock = threading.Lock()

  def register(self, filesystem_cls: Type[filesystem.Filesystem],
               priority: int, use_as_fallback: bool = False) -> None:
    """Register a filesystem implementation.

    Args:
      filesystem_cls: Subclass of `tfx.dsl.io.filesystem.Filesystem`.
      priority: Integer priority index (lower is more preferred) specifying
        plugin search order for filesystem schemes supported by the filesystem
        class.
      use_as_fallback: If `True`, use the given filesystem class to provide
        fallback access to filesystem not explicitly supported by a registered
        filesystem plugin.
    """
    with self._registration_lock:
      self._filesystem_priority[filesystem_cls] = priority
      for scheme in filesystem_cls.SUPPORTED_SCHEMES:
        current_preferred = self._preferred_filesystem_by_scheme.get(scheme)
        if (not current_preferred or
            priority < self._filesystem_priority[current_preferred]):
          self._preferred_filesystem_by_scheme[scheme] = filesystem_cls
      if use_as_fallback:
        if (not self._fallback_filesystem or
            priority < self._filesystem_priority[self._fallback_filesystem]):
          self._fallback_filesystem = filesystem_cls

  def get_filesystem_for_scheme(
      self, scheme: PathType) -> Type[filesystem.Filesystem]:
    """Get filesystem plugin for given scheme string."""
    if isinstance(scheme, bytes):
      scheme = scheme.decode('utf-8')
    if scheme not in self._preferred_filesystem_by_scheme:
      if self._fallback_filesystem:
        return self._fallback_filesystem
      else:
        raise Exception(
            ('The filesystem scheme %r is not available for use. For expanded '
             'filesystem scheme support, install the `tensorflow` package to '
             'enable additional filesystem plugins.') % scheme)
    return self._preferred_filesystem_by_scheme[scheme]

  def get_filesystem_for_path(self,
                              path: PathType) -> Type[filesystem.Filesystem]:
    """Get filesystem plugin for given path."""
    # Assume local path by default, but extract filesystem prefix if available.
    if isinstance(path, str):
      path_bytes = path.encode('utf-8')
    elif isinstance(path, bytes):
      path_bytes = path
    else:
      raise ValueError('Invalid path type: %r.' % path)
    result = re.match(b'^([a-z0-9]+://)', path_bytes)
    if result:
      scheme = result.group(1).decode('utf-8')
    else:
      scheme = ''
    return self.get_filesystem_for_scheme(scheme)


# Default global instance of the filesystem registry.
DEFAULT_FILESYSTEM_REGISTRY = FilesystemRegistry()
