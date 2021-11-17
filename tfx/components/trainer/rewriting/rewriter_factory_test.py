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
"""Tests for third_party.tfx.components.trainer.rewriting.rewriter_factory."""

import importlib
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from tfx.components.trainer.rewriting import rewriter_factory


def _tfjs_installed():
  try:
    importlib.import_module('tensorflowjs')
  except ImportError:
    return False
  else:
    return True


class RewriterFactoryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('TFLite', rewriter_factory.TFLITE_REWRITER))
  def testRewriterFactorySuccessfullyCreated(self, rewriter_name):
    tfrw = rewriter_factory.create_rewriter(rewriter_name, name='my_rewriter')
    self.assertTrue(tfrw)
    self.assertEqual(type(tfrw).__name__, rewriter_name)
    self.assertEqual(tfrw.name, 'my_rewriter')

  @unittest.skipUnless(_tfjs_installed(), 'tensorflowjs is not installed')
  def testRewriterFactorySuccessfullyCreatedTFJSRewriter(self):
    tfrw = rewriter_factory.create_rewriter(rewriter_factory.TFJS_REWRITER,
                                            name='my_rewriter')
    self.assertTrue(tfrw)
    self.assertEqual(type(tfrw).__name__, rewriter_factory.TFJS_REWRITER)
    self.assertEqual(tfrw.name, 'my_rewriter')

if __name__ == '__main__':
  absltest.main()
