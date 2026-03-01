# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Module for EqualPropertyValues operator."""

from typing import Sequence, Union

from absl import logging
from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import json_utils


class EqualPropertyValues(
    resolver_op.ResolverOp,
    canonical_name='tfx.EqualPropertyValues',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """EqualPropertyValues operator."""

  # The property key to match by.
  property_key = resolver_op.Property(type=str)

  # The expected property value to match by.
  property_value = resolver_op.Property(
      type=Union[int, float, bool, str, json_utils.JsonValue]
  )

  # If True, match by the artifact's custom properties, else match by its
  # properties. Defaults to True.
  is_custom_property = resolver_op.Property(type=bool, default=True)

  def apply(
      self,
      input_list: Sequence[types.Artifact],
  ) -> Sequence[types.Artifact]:
    """Returns artifacts with matching custom property (or property) values."""
    output_artifact_list = []
    for artifact in input_list:
      if self.is_custom_property:
        if not artifact.has_custom_property(self.property_key):
          logging.warning(
              'The artifact %s does not contain the custom property %s.',
              artifact,
              self.property_key,
          )
          continue
        actual_property_value = artifact.get_custom_property(self.property_key)
      else:
        if not artifact.has_property(self.property_key):
          logging.warning(
              'The artifact %s does not contain the property %s.',
              artifact,
              self.property_key,
          )
          continue
        actual_property_value = getattr(artifact, self.property_key)

      if actual_property_value == self.property_value:
        output_artifact_list.append(artifact)

    return output_artifact_list
