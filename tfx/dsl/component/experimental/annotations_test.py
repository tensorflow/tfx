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
from tfx.dsl.component.experimental import annotations_test_proto_pb2
from tfx.types import artifact
from tfx.types import standard_artifacts
from tfx.types import value_artifact


class AnnotationsTest(tf.test.TestCase):

  def testArtifactGenericAnnotation(self):
    # Error: type hint whose parameter is not an Artifact subclass.
    with self.assertRaisesRegex(
        ValueError, 'expects .* a concrete subclass of'
    ):
      _ = annotations._ArtifactGeneric[int]  # pytype: disable=unsupported-operands

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegex(
        ValueError, 'expects .* a concrete subclass of'
    ):
      _ = annotations._ArtifactGeneric[artifact.Artifact]

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegex(
        ValueError, 'expects .* a concrete subclass of'
    ):
      _ = annotations._ArtifactGeneric[value_artifact.ValueArtifact]

    # OK.
    _ = annotations._ArtifactGeneric[standard_artifacts.Examples]

  def testArtifactAnnotationUsage(self):
    _ = annotations.InputArtifact[standard_artifacts.Examples]
    _ = annotations.OutputArtifact[standard_artifacts.Examples]
    _ = annotations.AsyncOutputArtifact[standard_artifacts.Model]

  def testPrimitivAndProtoTypeGenericAnnotation(self):
    # Error: type hint whose parameter is not a primitive or a proto type
    # pytype: disable=unsupported-operands
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric[artifact.Artifact]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric[object]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric[123]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric['string']
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric[Dict[int, int]]
    with self.assertRaisesRegex(
        ValueError, 'T to be `int`, `float`, `str`, `bool`'
    ):
      _ = annotations._PrimitiveAndProtoTypeGeneric[bytes]
    # pytype: enable=unsupported-operands
    # OK.
    _ = annotations._PrimitiveAndProtoTypeGeneric[int]
    _ = annotations._PrimitiveAndProtoTypeGeneric[float]
    _ = annotations._PrimitiveAndProtoTypeGeneric[str]
    _ = annotations._PrimitiveAndProtoTypeGeneric[bool]
    _ = annotations._PrimitiveAndProtoTypeGeneric[Dict[str, float]]
    _ = annotations._PrimitiveAndProtoTypeGeneric[bool]
    _ = annotations._PrimitiveAndProtoTypeGeneric[
        annotations_test_proto_pb2.TestMessage
    ]

  def testPipelineTypeGenericAnnotation(self):
    # Error: type hint whose parameter is not a primitive type
    with self.assertRaisesRegex(ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[artifact.Artifact]
    with self.assertRaisesRegex(ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[object]
    # pytype: disable=unsupported-operands
    with self.assertRaisesRegex(ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric[123]
    with self.assertRaisesRegex(ValueError, 'T to be `beam.Pipeline`'):
      _ = annotations._PipelineTypeGeneric['string']
    # pytype: enable=unsupported-operands

    # OK.
    _ = annotations._PipelineTypeGeneric[beam.Pipeline]

  def testParameterUsage(self):
    _ = annotations.Parameter[int]
    _ = annotations.Parameter[float]
    _ = annotations.Parameter[str]
    _ = annotations.Parameter[bool]
    _ = annotations.Parameter[annotations_test_proto_pb2.TestMessage]
