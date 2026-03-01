# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Module for Unnest operator."""

from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.utils import typing_utils


class Unnest(
    resolver_op.ResolverOp,
    canonical_name='tfx.internal.Unnest',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """Unnest operator.

  Unnest operator split a *`key` channel* of multiple artifacts into multiple
  dicts each with a channel with single artifact. Pseudo code example:

      Unnest({x: [x1, x2, x3]}, key=x)
        = [{x: [x1]}, {x: [x2]}, {x: [x3]}]

  For channels other than key channel remains the same. Pseudo code example:

      Unnest({x: [x1, x2, x3], y: [y1]}, key=x)
        = [{x: [x1], y: [y1]}, {x: [x2], y: [y1]}, {x: [x3], y: [y1]}]
  """
  key = resolver_op.Property(type=str)

  def apply(self, input_dict: typing_utils.ArtifactMultiMap):
    if self.key not in input_dict:
      raise exceptions.FailedPreconditionError(
          f'Input dict does not contain the key {self.key}. '
          f'Available: {list(input_dict.keys())}')

    main_channel = input_dict.get(self.key)
    rest = {k: v for k, v in input_dict.items() if k != self.key}
    result = []
    for main_artifact in main_channel:
      result.append({self.key: [main_artifact], **rest})
    return result
