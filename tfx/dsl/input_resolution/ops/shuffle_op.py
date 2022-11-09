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
"""Module for Shuffle operator."""

import random

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op


class Shuffle(
    resolver_op.ResolverOp,
    canonical_name='tfx.Shuffle',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """Shuffle operator."""

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns the artifacts in a random order."""
    # We use sample() becuase input_list is non-mutable and shuffle() modifies
    # the list in place.
    return random.sample(input_list, len(input_list))
