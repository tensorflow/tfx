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
import types
from typing import Any, Callable, Text, Type

import six

from tfx.utils import io_utils


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


def import_func_from_source(source_path: Text, fn_name: Text) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file."""

  # If module path is not local, download to local file-system first,
  # because importlib can't import from GCS
  source_path = io_utils.ensure_local(source_path)

  try:
    if six.PY2:
      import imp  # pylint: disable=g-import-not-at-top
      try:
        user_module = imp.load_source('user_module', source_path)
        return getattr(user_module, fn_name)
      except IOError:
        raise

    else:
      loader = importlib.machinery.SourceFileLoader(
          fullname='user_module',
          path=source_path,
      )
      user_module = types.ModuleType(loader.name)
      loader.exec_module(user_module)
      return getattr(user_module, fn_name)

  except IOError:
    raise ImportError('{} in {} not found in import_func_from_source()'.format(
        fn_name, source_path))


def import_func_from_module(module_path: Text, fn_name: Text) -> Callable:  # pylint: disable=g-bare-generic
  """Imports a function from a module provided as source file or module path."""
  user_module = importlib.import_module(module_path)
  return getattr(user_module, fn_name)
