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
"""Module for ListToDict operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils


class ListToDict(
    resolver_op.ResolverOp,
    canonical_name='tfx.ListToDict',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """ListToDict operator."""

  # The number of keys in the dictionary. If n < 0, then n will be set to the
  # length of input_list. If n > len(input_list), then the keys from
  # "len(input_list)", ..., "n - 1" will have empty lists as values. If n <
  # len(input_list), then only the first n artifacts in input_list will be
  # considered.
  n = resolver_op.Property(type=int, default=-1)

  def apply(
      self, input_list: Sequence[types.Artifact]
  ) -> typing_utils.ArtifactMultiMap:
    """Returns the list of artifacts as a dict, sorted by creation time.

    For example, given artifacts [a_0, a_1, a_2], ListToDict will return
    {'0': [a_0], '1': [a_1], '2': [a_2]}.

    Args:
      input_list: The list of artifact.

    Returns:
      A dict of artifacts, with '0', '1', ... as keys and a single artifact
      wrapped in a list as values.
    """
    if not input_list:
      return {}

    if self.n == -1:
      num_keys = len(input_list)
    else:
      num_keys = self.n

    output_dict = {}
    for i in range(num_keys):
      if i < len(input_list):
        output_dict[str(i)] = [input_list[i]]
      else:
        output_dict[str(i)] = []
    return output_dict
