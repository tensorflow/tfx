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
"""Common functionalities used in transform executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Text, Sequence, Mapping


def GetValues(inputs,
              label):
  """Retrieves the value of the given labeled input.

  Args:
    inputs: Dict from label to a value list.
    label: Label of the value to retrieve.

  Returns:
    A list of values, or empty list if there's no value.

  Raises:
    ValueError: If label is not one of the valid input labels.
  """
  if label not in inputs:
    return []
  values = inputs.get(label)
  if not isinstance(values, list):
    return [values]
  return values


def GetSoleValue(inputs, label,
                 strict=True):
  """Helper method for retrieving a sole labeled input.

  Args:
    inputs: Dict from label to a value list.
    label: Label of the value to retrieve.
    strict: If true, exact one value should exist for label.

  Returns:
    A sole labeled value.

  Raises:
    ValueError: If there is no/multiple input associated with the label.
  """
  values = GetValues(inputs, label)
  if len(values) > 1:
    raise ValueError(
        'There should not be more than one value for label {}'.format(label))
  if strict:
    if len(values) != 1:
      raise ValueError(
          'There should be one and only one value for label {}'.format(label))
  else:
    if not values:
      return None
  return values[0]
