# Lint as: python2, python3
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
"""Tests for tfx.components.base.annotations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Text

# Standard Imports

import tensorflow as tf

from tfx.dsl.component.experimental import annotations
from tfx.types import artifact
from tfx.types import standard_artifacts


class AnnotationsTest(tf.test.TestCase):

  def testArtifactGenericAnnotation(self):
    # Error: type hint whose parameter is not an Artifact subclass.
    with self.assertRaisesRegexp(ValueError,
                                 'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[int]

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegexp(ValueError,
                                 'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[artifact.Artifact]

    # Error: type hint with abstract Artifact subclass.
    with self.assertRaisesRegexp(ValueError,
                                 'expects .* a concrete subclass of'):
      _ = annotations._ArtifactGeneric[artifact.ValueArtifact]

    # OK.
    _ = annotations._ArtifactGeneric[standard_artifacts.Examples]

  def testArtifactAnnotationUsage(self):
    _ = annotations.InputArtifact[standard_artifacts.Examples]
    _ = annotations.OutputArtifact[standard_artifacts.Examples]

  def testPrimitiveTypeGenericAnnotation(self):
    # Error: type hint whose parameter is not a primitive type
    with self.assertRaisesRegexp(ValueError,
                                 'T to be `int`, `float`, `str` or `bytes`'):
      _ = annotations._PrimitiveTypeGeneric[artifact.Artifact]
    with self.assertRaisesRegexp(ValueError,
                                 'T to be `int`, `float`, `str` or `bytes`'):
      _ = annotations._PrimitiveTypeGeneric[object]
    with self.assertRaisesRegexp(ValueError,
                                 'T to be `int`, `float`, `str` or `bytes`'):
      _ = annotations._PrimitiveTypeGeneric[123]
    with self.assertRaisesRegexp(ValueError,
                                 'T to be `int`, `float`, `str` or `bytes`'):
      _ = annotations._PrimitiveTypeGeneric['string']

    # OK.
    _ = annotations._PrimitiveTypeGeneric[int]
    _ = annotations._PrimitiveTypeGeneric[float]
    _ = annotations._PrimitiveTypeGeneric[Text]
    _ = annotations._PrimitiveTypeGeneric[bytes]

  def testParameterUsage(self):
    _ = annotations.Parameter[int]
    _ = annotations.Parameter[float]
    _ = annotations.Parameter[Text]
    _ = annotations.Parameter[bytes]


if __name__ == '__main__':
  tf.test.main()
