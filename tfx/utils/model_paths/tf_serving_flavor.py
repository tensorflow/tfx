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
"""Module for constructing and parsing tensorflow-serving-flavored model path.

In TensorFlow Serving, the model should be stored with the directory structure
`{model_base_path}/{model_name}/{version}` to be correctly recognized by the
model server. We call this a *TFS-flavored model path*.

Example:

```
/foo/bar/        # A `model_base_path`
  my_model/      # A `model_name`
    1582072718/  # An integer `version`
      (Your exported SavedModel)
```

If you're using TensorFlow estimator API, the exporter uses this directory
structure to organize the model, where `model_name` is the name of the exporter.

TensorFlow Serving also requires the version segment to be an integer (mostly
a unix timestamp) to track the latest model easily.
"""

import os
from typing import Optional, Tuple


def make_model_path(model_base_path: str, model_name: str,
                    version: int) -> str:
  """Make a TFS-flavored model path.

  Args:
    model_base_path: A base path containing the directory of model_name.
    model_name: A name of the model.
    version: An integer version of the model.

  Returns:
    `{model_base_path}/{model_name}/{version}`.
  """
  return os.path.join(model_base_path, model_name, str(version))


def parse_model_path(
    model_path: str,
    expected_model_name: Optional[str] = None) -> Tuple[str, str, int]:
  """Parse model_path into parts of TFS flavor.

  Args:
    model_path: A TFS-flavored model path.
    expected_model_name: Expected model_name as defined from the module
        docstring. If model_name does not match, parsing will be failed.

  Raises:
    ValueError: If model path is invalid (not TFS-flavored).

  Returns:
    Tuple of (model_base_path, model_name, version)
  """
  rest, version = os.path.split(model_path)
  if not rest:
    raise ValueError('model_path is too short ({})'.format(model_path))
  if not version.isdigit():
    raise ValueError('No version segment ({})'.format(model_path))
  version = int(version)

  model_base_path, model_name = os.path.split(rest)
  if expected_model_name is not None and model_name != expected_model_name:
    raise ValueError('model_name does not match (expected={}, actual={})'
                     .format(expected_model_name, model_path))

  return model_base_path, model_name, version


def parse_model_base_path(model_path: str) -> str:
  """Parse model_base_path from the TFS-flavored model path.

  Args:
    model_path: A TFS-flavored model path.

  Raises:
    ValueError: If model path is invalid (not TFS-flavored).

  Returns:
    model_base_path as defined from the module docstring.
  """
  return parse_model_path(model_path)[0]
