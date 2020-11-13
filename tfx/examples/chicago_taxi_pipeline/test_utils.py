# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Utilities for taxi E2E tests."""
import os
from typing import Iterable

import tensorflow as tf
from tfx.dsl.io import fileio


class TaxiTest(tf.test.TestCase):
  """Convenient class wrapping methods shared for taxi E2E tests."""

  def assertComponentsExecuted(self, pipeline_root: str,
                               components: Iterable[str]) -> None:
    for component_name in components:
      component_path = os.path.join(pipeline_root, component_name)
      self.assertTrue(fileio.exists(component_path))
