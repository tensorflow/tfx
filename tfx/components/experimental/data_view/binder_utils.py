# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Utilities to manage data view information."""

from typing import Optional, Tuple

from tfx import types
from tfx.components.experimental.data_view import constants


def bind_data_view_to_artifact(data_view: types.Artifact,
                               examples: types.Artifact) -> None:
  examples.set_string_custom_property(
      constants.DATA_VIEW_CREATE_TIME_KEY,
      str(data_view.mlmd_artifact.create_time_since_epoch))
  examples.set_string_custom_property(constants.DATA_VIEW_URI_PROPERTY_KEY,
                                      data_view.uri)


def get_data_view_info(
    examples: types.Artifact) -> Optional[Tuple[str, int]]:
  """Returns the payload format and data view URI and ID from examples."""
  data_view_uri = examples.get_string_custom_property(
      constants.DATA_VIEW_URI_PROPERTY_KEY)
  if not data_view_uri:
    return None

  assert examples.has_custom_property(constants.DATA_VIEW_CREATE_TIME_KEY)
  # The creation time could be an int or str. Legacy artifacts will contain
  # an int custom property.
  data_view_create_time = examples.get_custom_property(
      constants.DATA_VIEW_CREATE_TIME_KEY)
  data_view_create_time = int(data_view_create_time)
  return data_view_uri, data_view_create_time
