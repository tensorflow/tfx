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
"""Tests for tfx.components.base.container_component."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import unittest

# Standard Imports

import six

import tensorflow as tf

from tfx.components.base import container_component
from tfx.components.base.container_component import container_component_from_typehints
from tfx.types.annotations import ComponentOutput


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class ContainerComponentTest(tf.test.TestCase):

  def testStubTemplate(self):
    template = container_component._get_stub_template()
    self.assertNotIn('# BEGIN_STUB', template)
    self.assertNotIn('# END_STUB', template)
    self.assertIn('# INSERTION_POINT', template)

  def testComponentConstruction(self):

    @container_component_from_typehints('my/docker:image')
    def injector() -> ComponentOutput(a=int, b=int, c=Text, d=bytes):
      return {'a': 10, 'b': 22, 'c': 'unicode', 'd': b'bytes'}

    @container_component_from_typehints('my/docker:image')
    def simple_component(  # pylint: disable=unused-argument
        a: int, b: int, c: Text, d: bytes) -> ComponentOutput(e=float):
      return {'e': float(a + b)}

    instance_1 = injector()
    unused_instance_2 = simple_component(
        a=instance_1.outputs['a'],
        b=instance_1.outputs['b'],
        c=instance_1.outputs['c'],
        d=instance_1.outputs['d'])

    print('simple_component', simple_component)
    print('image', simple_component.EXECUTOR_SPEC.image)
    print('command', simple_component.EXECUTOR_SPEC.command)
    print('args', simple_component.EXECUTOR_SPEC.args)


if __name__ == '__main__':
  tf.test.main()
