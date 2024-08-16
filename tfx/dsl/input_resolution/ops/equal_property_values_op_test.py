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
"""Tests for tfx.dsl.input_resolution.ops.equal_property_values_op."""

import tensorflow as tf
from tfx import v1 as tfx
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.types import artifact as tfx_artifact


class EqualPropertyValuesOpTest(tf.test.TestCase):

  def _equal_property_values(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.EqualPropertyValues, args=args, kwargs=kwargs
    )

  def create_artifacts(self, n: int):
    artifacts = []
    for i in range(1, n + 1):
      dummy_artifact = DummyArtifact()
      dummy_artifact.num_steps = i
      artifacts.append(dummy_artifact)
    return artifacts

  def test_equal_property_values_no_property(self):
    artifacts = self.create_artifacts(n=3)
    results = self._equal_property_values(
        artifacts, property_key="id", property_value=3
    )
    self.assertEmpty(results)

  def test_equal_property_values_not_equal(self):
    artifacts = self.create_artifacts(n=3)
    results = self._equal_property_values(
        artifacts, property_key="num_steps", property_value=4)
    self.assertEmpty(results)

  def test_equal_property_values_has_equal_property(self):
    artifacts = self.create_artifacts(n=3)
    with self.subTest(name="HasProperty"):
      results = self._equal_property_values(
          artifacts, property_key="num_steps", property_value=3)
      self.assertLen(results, 1)
      self.assertEqual(results[0].num_steps, 3)
    with self.subTest(name="MoreThanOneHasProperty"):
      new_artifact = DummyArtifact()
      new_artifact.num_steps = 3
      artifacts.append(new_artifact)
      results = self._equal_property_values(
          artifacts, property_key="num_steps", property_value=3,
          is_custom_property=True)
      self.assertLen(results, 2)
      self.assertEqual(results[0].num_steps, results[1].num_steps)

  def test_property_and_custom_property_mismatch(self):
    artifact_no_custom = DummyArtifactNoCustomArtifact()
    artifact_no_custom.num_steps = 1
    with self.subTest(name="NoCustomProperty"):
      # Checking non-custom property will return the correct value.
      results = self._equal_property_values(
          [artifact_no_custom], property_key="num_steps", property_value=1
      )
      self.assertEmpty(results)
    with self.subTest(name="HasNonCustomProperty"):
      results = self._equal_property_values(
          [artifact_no_custom], property_key="num_steps", property_value=1,
          is_custom_property=False)
      self.assertLen(results, 1)
      self.assertEqual(results[0].num_steps, 1)


class DummyArtifact(tfx.dsl.Artifact):
  TYPE_NAME = "DummyArtifact"

  @property
  def num_steps(self):
    return self.get_int_custom_property("num_steps")

  @num_steps.setter
  def num_steps(self, value):
    self.set_int_custom_property("num_steps", value)


class DummyArtifactNoCustomArtifact(tfx.dsl.Artifact):
  TYPE_NAME = "DummyArtifactNoCustomArtifact"
  PROPERTIES = {
      "num_steps": tfx_artifact.Property(type=tfx_artifact.PropertyType.INT),
  }
