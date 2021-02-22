# Lint as: python2, python3
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
"""TFX type definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import sys
import threading
from typing import Any, Callable, Text, Type

from absl import logging
from tfx.utils import io_utils

_imported_modules_from_source = {}
_imported_modules_from_source_lock = threading.Lock()


def import_class_by_path(class_path: Text) -> Type[Any]:
  """Import a class by its <module>.<name> path.

  Args:
    class_path: <module>.<name> for a class.

  Returns:
    Class object for the given class_path.
  """
  classname = class_path.split('.')[-1]
  modulename = '.'.join(class_path.split('.')[0:-1])
  mod = importlib.import_module(modulename)
  return getattr(mod, classname)


# TODO(b/175174419): Revisit the workaround for multiple invocations of
# import_func_from_source.
def import_func_from_source(source_path: Text, fn_name: Text) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file."""

  # If module path is not local, download to local file-system first,
  # because importlib can't import from GCS
  source_path = io_utils.ensure_local(source_path)

  with _imported_modules_from_source_lock:
    if source_path not in _imported_modules_from_source:
      logging.info('Loading %s because it has not been loaded before.',
                   source_path)
      # Create a unique module name.
      module_name = 'user_module_%d' % len(_imported_modules_from_source)
      try:
        loader = importlib.machinery.SourceFileLoader(
            fullname=module_name,
            path=source_path,
        )
        spec = importlib.util.spec_from_loader(
            loader.name, loader, origin=source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = module
        loader.exec_module(module)
        sys.meta_path.append(_ModuleFinder({module_name: source_path}))
        _imported_modules_from_source[source_path] = module
      except IOError:
        raise ImportError('{} in {} not found in '
                          'import_func_from_source()'.format(
                              fn_name, source_path))
    else:
      logging.info('%s is already loaded.', source_path)
  return getattr(_imported_modules_from_source[source_path], fn_name)


def import_func_from_module(module_path: Text, fn_name: Text) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file or module path."""
  user_module = importlib.import_module(module_path)
  return getattr(user_module, fn_name)


class _ModuleFinder(importlib.abc.MetaPathFinder):
  """Registers custom modules for Interactive Context."""

  def __init__(self, path_map: dict):  # pylint: disable=g-bare-generic
    self.path_map = path_map

  def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
    if fullname not in self.path_map:
      return None
    return importlib.util.spec_from_file_location(
        fullname, self.path_map[fullname])

  def find_module(self, fullname, path):
    pass
