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
"""Utilities for DSL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text
from tfx.utils import types


def csv_input(uri):
  """Helper function to declare input for csv_example_gen component.

  Args:
    uri: path of an external directory with a single csv file inside.

  Returns:
    A list of TfxType which will be constructed as Channel later.
  """
  instance = types.TfxType(type_name='ExternalPath')
  instance.uri = uri
  return [instance]
