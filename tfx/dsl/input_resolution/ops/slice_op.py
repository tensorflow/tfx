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
"""Module for Slice operator."""

from typing import Optional, Sequence

from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.types import artifact


class Slice(
    resolver_op.ResolverOp,
    canonical_name='tfx.Slice',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """Slice operator similar to python slice.

  For those who are not familiar with python slicing, python list can have a
  sliced index in the form of `values[start:stop]` to select the subitems of the
  list. `start` or `stop` can be omitted (e.g. `values[:3]`, `values[1:]`, or
  even `values[:]`), have negative values (e.g. `values[-2:]` selects the last 2
  elements).

  There is also a `step` argument that can be specified as part of the slice,
  which controls the step size, or the stride, of the slice range, but this is
  omitted in the Slice operator for brevity.

  Usage:

  ```python
  @resolver_function
  def head(artifacts, n: int):
    return Slice(artifacts, stop=n, min_count=n)
  ```

  Attributes:
    start: A start index (inclusive) of the slice range. Can be negative (index
      from the backward), or omitted (range from the beginning).
    stop: A stop index (exclusive) of the slice range. Can be negative (index
      from the backward), or omitted (range to the end).
    step: A step value of the slice. Can be negative (step backward). By default
      it is considered 1.
    min_count: If specified, the operator ensures the sliced range contains at
      least this number of items. If min_count is not specified, the operator
      would raise an InsufficientInputError, which in synchronous pipeline
      treated as an error, and in asynchronous pipeline treated as an idle thus
      wait until min_count is met.
  """

  start = resolver_op.Property(type=Optional[int], default=None)
  stop = resolver_op.Property(type=Optional[int], default=None)
  step = resolver_op.Property(type=Optional[int], default=None)
  min_count = resolver_op.Property(type=int, default=0)

  def apply(self, artifacts: Sequence[artifact.Artifact]):
    # Note: `values[None:3]` is equivalent to `values[:3]`, so we can safely
    # use `None` value of the operator property.
    result = artifacts[self.start : self.stop : self.step]
    if 0 <= len(result) < self.min_count:
      # InsufficientInputError will be treated ERROR in sync pipeline, but
      # IDLE in async pipeline.
      raise exceptions.InsufficientInputError(
          f'slice[{self.start}:{self.stop}:{self.stop}] on list of length'
          f' {len(artifacts)} has {len(result)} items which is less than'
          f' min_count = {self.min_count}.'
      )
    return result
