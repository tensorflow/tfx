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

import importlib
import sys
import threading
from typing import Any, Callable, Type

from absl import logging
from tfx.utils import io_utils


def import_class_by_path(class_path: str) -> Type[Any]:
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


def import_func_from_module(module_path: str, fn_name: str) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file or module path."""
  original_module_path = module_path
  wheel_context_manager = None
  if '@' in module_path:
    # The module path is a combined specification of a module path in a wheel
    # file path.
    module_path, wheel_path = module_path.split('@', maxsplit=1)
    wheel_path = io_utils.ensure_local(wheel_path)
    # Install pip dependencies and add it to the import resolution path.
    # TODO(b/187122070): Move `udf_utils.py` to `tfx/utils` and fix circular
    # dependency.
    from tfx.components.util import udf_utils  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
    wheel_context_manager = udf_utils.TempPipInstallContext([wheel_path])
    with _imported_modules_from_source_lock:
      wheel_context_manager.__enter__()
  if module_path in sys.modules:
    importlib.reload(sys.modules[module_path])
  try:
    user_module = importlib.import_module(module_path)
  except ImportError as e:
    raise ImportError('Could not import requested module path %r.' %
                      original_module_path) from e
  # Restore original sys.path.
  if wheel_context_manager:
    with _imported_modules_from_source_lock:
      wheel_context_manager.__exit__(None, None, None)
  return getattr(user_module, fn_name)


# This lock is used both for access to class variables for _TfxModuleFinder
# and usage of that class, therefore must be RLock
# to avoid deadlock among multiple levels of call stack.
_imported_modules_from_source_lock = threading.RLock()


class _TfxModuleFinder(importlib.abc.MetaPathFinder):
  """Registers custom modules for Interactive Context."""

  _modules = {}  # a mapping fullname -> source_path

  def find_spec(self, fullname, path, target=None):
    del path
    del target
    with _imported_modules_from_source_lock:
      if fullname not in self._modules:
        return None
      source_path = self._modules[fullname]
      loader = importlib.machinery.SourceFileLoader(
          fullname=fullname, path=source_path)
      return importlib.util.spec_from_loader(
          fullname, loader, origin=source_path)

  def find_module(self, fullname, path):
    pass

  def register_module(self, fullname, path):
    """Registers and imports a new module."""
    with _imported_modules_from_source_lock:
      if fullname in self._modules:
        raise ValueError('Module %s is already registered' % fullname)
      self._modules[fullname] = path

  def get_module_name_by_path(self, path):
    with _imported_modules_from_source_lock:
      for module_name, source_path in self._modules.items():
        if source_path == path:
          return module_name
      return None

  @property
  def count_registered(self):
    with _imported_modules_from_source_lock:
      return len(self._modules)


_tfx_module_finder = _TfxModuleFinder()
with _imported_modules_from_source_lock:
  sys.meta_path.append(_tfx_module_finder)


# TODO(b/175174419): Revisit the workaround for multiple invocations of
# import_func_from_source.
def import_func_from_source(source_path: str, fn_name: str) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file."""

  # If module path is not local, download to local file-system first,
  # because importlib can't import from GCS
  source_path = io_utils.ensure_local(source_path)

  module = None
  with _imported_modules_from_source_lock:
    if _tfx_module_finder.get_module_name_by_path(source_path) is None:
      # Create a unique module name.
      module_name = 'user_module_%d' % _tfx_module_finder.count_registered
      logging.info(
          'Loading source_path %s as name %s '
          'because it has not been loaded before.', source_path, module_name)
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
        _tfx_module_finder.register_module(module_name, source_path)
      except IOError:
        raise ImportError('{} in {} not found in '
                          'import_func_from_source()'.format(
                              fn_name, source_path))
    else:
      logging.info('%s is already loaded, reloading', source_path)
      module_name = _tfx_module_finder.get_module_name_by_path(source_path)
      module = sys.modules[module_name]
      importlib.reload(module)
  return getattr(module, fn_name)
