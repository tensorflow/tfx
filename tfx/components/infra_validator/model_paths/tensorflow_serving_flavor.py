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
"""Module for constructing and parsing tensorflow-serving-flavored model path.

In TensorFlow Serving, model should be stored under directory structure
`{base_path}/{model_name}/{version:int}` to be correctly recognized from the
model server. We call it tensorflow-serving-flavored (or TFS-flavored) model
path.

```
/foo/bar/        # base_path
  my_model/      # model_name
    1582072718/  # version
      (Your exported model)
```

If you're using TensorFlow estimator API, exporter use this directory structure
to organize the model, where `model_name` is the name of the exporter.

TensorFlow Serving also requires the version segment to be an integer (mostly
a unix timestamp) to track the latest model easily.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Optional, Text, Tuple


# TODO(b/149534564): Replace os.path to pathlib after py2 deprecation.
def make_model_path(base_path: Text, model_name: Text, version: int) -> Text:
  """Make a TFS-flavored model path.

  Args:
    base_path: A base path containig the directory of model_name.
    model_name: A name of the model.
    version: An integer version of the model.

  Returns:
    `{base_path}/{model_name}/{version}`.
  """
  if isinstance(version, str) and not version.isdigit():
    raise ValueError('version should be an integer, got {}'.format(version))
  return os.path.join(base_path, model_name, str(version))


def parse_model_path(
    model_path: Text,
    expected_model_name: Optional[Text] = None) -> Tuple[Text, Text, int]:
  """Parse model_path into parts of TFS flavor.

  Args:
    model_path: An TFS-flavored model path.
    expected_model_name: Expected model_name as defined from the module
        docstring. If model_name does not match, parse will be failed.

  Raises:
    ValueError: If model path is invalid (not TFS-flavored).

  Returns:
    Tuple of (base_path, model_name, version)
  """
  rest, version = os.path.split(model_path)
  if not rest:
    raise ValueError('model_path is too short ({})'.format(model_path))
  if not version or not version.isdigit():
    raise ValueError('No version segment ({})'.format(model_path))
  version = int(version)

  base_path, model_name = os.path.split(rest)
  if expected_model_name is not None and model_name != expected_model_name:
    raise ValueError('model_name does not match (expected={}, actual={})'
                     .format(expected_model_name, model_path))

  return base_path, model_name, version


def parse_base_path(model_path: Text) -> Text:
  """Parse base_path from the TFS-flavored model path.

  Args:
    model_path: An TFS-flavored model path.

  Raises:
    ValueError: If model path is invalid (not TFS-flavored).

  Returns:
    base_path as defined from the module docstring.
  """
  return parse_model_path(model_path)[0]
