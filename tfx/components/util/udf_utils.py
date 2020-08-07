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
"""Utilies for user defined functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Dict, Optional, Text

from tfx.utils import import_utils

# Key for module file.
_MODULE_FILE_KEY = 'module_file'


# TODO(b/157155972): improve user code support.
def get_fn(exec_properties: Dict[Text, Any],
           fn_name: Text) -> Callable[..., Any]:
  """Loads and returns user-defined function."""

  has_module_file = bool(exec_properties.get(_MODULE_FILE_KEY))
  has_fn = bool(exec_properties.get(fn_name))

  if has_module_file:
    if has_fn:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], exec_properties[fn_name])
    else:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], fn_name)
  elif has_fn:
    fn_path_split = exec_properties[fn_name].split('.')
    return import_utils.import_func_from_module('.'.join(fn_path_split[0:-1]),
                                                fn_path_split[-1])
  else:
    raise ValueError(
        'Neither module file or user function have been supplied in `exec_properties`.'
    )


def try_get_fn(exec_properties: Dict[Text, Any],
               fn_name: Text) -> Optional[Callable[..., Any]]:
  """Loads and returns user-defined function if exists."""
  try:
    return get_fn(exec_properties, fn_name)
  except (ValueError, AttributeError):
    # ValueError: module file or user function is unset.
    # AttributeError: the function doesn't exist in the module.
    return None
