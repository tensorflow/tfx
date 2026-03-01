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
"""Module for ExcludeSpans operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils


class ExcludeSpans(
    resolver_op.ResolverOp,
    canonical_name='tfx.ExcludeSpans',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """ExcludeSpans operator."""

  # The span numbers to exclude.
  denylist = resolver_op.Property(type=Sequence[int], default=[])

  def apply(
      self,
      input_list: Sequence[types.Artifact],
  ) -> Sequence[types.Artifact]:
    """Returns artifacts with spans not in denylist.

    Corresponds to exclude_span_numbers in RangeConfig in TFX.

    For example, if the artifacts have spans [1, 2, 2, 4], and
    denylist = [1, 2], then only the artifact [4] will be returned.

    Args:
      input_list: The list of Artifacts to parse.

    Returns:
      Artifacts with spans not in denylist.
    """
    valid_artifacts = ops_utils.get_valid_artifacts(input_list,
                                                    ops_utils.SPAN_PROPERTY)

    # Only return artifacts that do not have spans in denylist.
    return [a for a in valid_artifacts if a.span not in set(self.denylist)]
