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
"""Module for SlidingWindow operator."""

from typing import Sequence, Mapping

from tfx import types
from tfx.dsl.input_resolution import resolver_op


class SlidingWindow(
    resolver_op.ResolverOp,
    canonical_name='tfx.SlidingWindow',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST):
  """SlidingWindow operator."""

  # The length of the sliding window, must be > 0.
  window_size = resolver_op.Property(type=int, default=1)

  # The output key for the dicts in the returned ARTIFACT_MULTIMAP_LIST.
  output_key = resolver_op.Property(type=str, default='window')

  def apply(
      self, input_list: Sequence[types.Artifact]
  ) -> Sequence[Mapping[str, Sequence[types.Artifact]]]:
    """Applies a sliding window of size n to the list of artifacts.

    For example, for artifacts [A, B, C, D] with n=2, then a sliding window of 2
    will be applied, producing [[A, B], [B, C], [C, D]]. The stride is set to 1
    by default.

    Note that what will actually be returned is a an ARTIFACT_MULTIMAP_LIST:
    [{"window": [A, B]}, {"window": [B, C]}, {"window": [C, D]}]. The output_key
    is set to "window" by default.

    This is because a type of ARTIFACT_LIST_LIST is not yet supported in the IR
    compilation. The dictionaries will have to be unnested in the resolver
    function SlidingWindow is called in.

    Args:
      input_list: The Artifacts to filter.

    Returns:
      The artifacts with the sliding window applied, in list of dictionaries
        format.
    """
    if self.window_size < 1:
      raise ValueError(
          f'sliding_window must be > 0, but was set to {self.window_size}.')

    if not input_list:
      return []

    num_windows = len(input_list) - self.window_size + 1
    windows = [input_list[i:(i + self.window_size)] for i in range(num_windows)]
    return [{self.output_key: window} for window in windows]
