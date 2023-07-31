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
from tfx import types
from tfx.dsl.input_resolution import resolver_op


class EqualPropertyValues(
    resolver_op.ResolverOp,
    canonical_name="tfx.EqualPropertyValues",
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """EqualPropertyValues operator.

  The will iterate through the artifact list and find the artifact(s) with the
  matching custom property (or property) values.
  """
  property_key = resolver_op.Property(type=str)
  property_value = resolver_op.Property(type=Union[int, float, bool, str])

  # If True, only search the artifact's custom properties, else search its
  # properties. Defaults to True.
  is_custom_property = resolver_op.Property(type=bool, default=True)

  def apply(
      self,
      input_list: Sequence[types.Artifact],
  ) -> Sequence[types.Artifact]:
    """Apply the EqualPropertyValues operator.

    Args:
      input_list: A list of input artifacts.
    Returns:
      A list of dict of artifacts.
    """
    output_artifact_list = []
    for artifact in input_list:
      if self.is_custom_property:
        property_value = artifact.get_custom_property(self.property_key)
        if not property_value:
          raise ValueError(
              f"The artifact does not contain the property {self.property_key}."
          )
      else:
        if not hasattr(artifact, self.property_key):
          raise ValueError(
              f"The artifact does not contain the property {self.property_key}."
          )
        property_value = getattr(artifact, self.property_key)

      if property_value == self.property_value:
        output_artifact_list.append(artifact)

    return output_artifact_list
