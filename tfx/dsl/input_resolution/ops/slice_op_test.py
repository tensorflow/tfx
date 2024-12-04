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
"""Tests for tfx.dsl.input_resolution.ops.slice_op."""

from absl.testing import parameterized
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.types import artifact


class MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyArtifact'
  PROPERTIES = {'x': artifact.Property(artifact.PropertyType.INT)}

  def __init__(self, x: int):
    super().__init__()
    self.x = x

  def __eq__(self, other):
    return isinstance(other, MyArtifact) and self.x == other.x


class SliceOpTest(test_utils.ResolverTestCase, parameterized.TestCase):
  _artifacts = [MyArtifact(i) for i in range(3)]

  def _slice(self, artifacts: list[artifact.Artifact], **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.Slice, args=(artifacts,), kwargs=kwargs
    )

  @parameterized.product(
      start=[-3, -2, -1, 0, 1, 2, 3, 4, None],
      stop=[-3, -2, -1, 0, 1, 2, 3, 4, None],
      step=[-2, -1, None, 1, 2],
  )
  def testSlice(self, start: int, stop: int, step: int):
    kwargs = {}
    if start is not None:
      kwargs['start'] = start
    if stop is not None:
      kwargs['stop'] = stop
    if step is not None:
      kwargs['step'] = step
    self.assertEqual(
        self._slice(self._artifacts, **kwargs), self._artifacts[start:stop:step]
    )

  def testSliceMinCount(self):
    inputs = self._artifacts[:1]
    with self.assertRaises(exceptions.InsufficientInputError):
      self._slice(inputs, start=1, stop=2, min_count=1)
