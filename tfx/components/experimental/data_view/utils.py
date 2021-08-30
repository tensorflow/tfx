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
"""Utilitiy functions related to DataView and Example artifacts."""
# TODO(zhuo): this file should be moved to tfx/components/util once DataView is
# no longer experimental.

from typing import Optional

from tfx import types
from tfx.components.experimental.data_view import constants
from tfx.types import standard_artifacts


def get_data_view_uri(examples: types.Artifact) -> Optional[str]:
  """Returns the URI to the DataView attached to an Examples artifact.

  Or None, if not attached.

  Args:
    examples: an Examples artifact.
  Returns:
    The URI to the DataView or None.
  """
  assert examples.type is standard_artifacts.Examples, (
      'examples must be of type standard_artifacts.Examples')
  data_view_uri = examples.get_string_custom_property(
      constants.DATA_VIEW_URI_PROPERTY_KEY)
  return data_view_uri if data_view_uri else None
