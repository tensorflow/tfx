# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.components.base.annotations."""

from typing import Dict

import apache_beam as beam
import tensorflow as tf
from tfx.dsl.component.experimental import annotations
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx.types import value_artifact


class AnnotationsTest(tf.test.TestCase):

  def testArtifactGenericAnnotation(self):
    # Error: type hint whose parameter is not an Artifact subclass.
    with self.assertRaisesRegex(ValueError,
                                'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[int]  # pytype: disable=unsupported-operands

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegex(ValueError,
                                'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[artifact.Artifact]

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegex(ValueError,
                                'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[value_artifact.ValueArtifact]

    # OK.
    _ = annotations._ArtifactGeneric[standard_artifacts.Examples]

  def testArtifactAnnotationUsage(self):
    _ = annotations.InputArtifact[standard_artifacts.Examples]
    _ = annotations.OutputArtifact[standard_artifacts.Examples]

  def testPrimitiveTypeGenericAnnotation(self):
    # Error: type hint whose parameter is not a primitive type
    # pytype: disable=unsupported-operands
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric[artifact.Artifact]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric[object]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric[123]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric['string']
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric[Dict[int, int]]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'):
      _ = annotations._PrimitiveTypeGeneric[bytes]
    # pytype: enable=unsupported-operands
    # OK.
    _ = annotations._PrimitiveTypeGeneric[int]
    _ = annotations._PrimitiveTypeGeneric[float]
    _ = annotations._PrimitiveTypeGeneric[str]
    _ = annotations._PrimitiveTypeGeneric[bool]
    _ = annotations._PrimitiveTypeGeneric[Dict[str, float]]
    _ = annotations._PrimitiveTypeGeneric[bool]

  def testPipelineTypeGenericAnnotation(self):
    # Error: type hint whose parameter is not a primitive type
    with self.assertRaisesRegex(
        ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[artifact.Artifact]
    with self.assertRaisesRegex(
        ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[object]
    # pytype: disable=unsupported-operands
    with self.assertRaisesRegex(
        ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[123]
    with self.assertRaisesRegex(
        ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric['string']
    # pytype: enable=unsupported-operands

    # OK.
    _ = annotations._PipelineTypeGeneric[beam.Pipeline]

  def testParameterUsage(self):
    _ = annotations.Parameter[int]
    _ = annotations.Parameter[float]
    _ = annotations.Parameter[str]
    _ = annotations.Parameter[bool]


if __name__ == '__main__':
  tf.test.main()
