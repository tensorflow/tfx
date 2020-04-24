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
"""Tests for third_party.tfx.components.trainer.rewriting.rewriter_factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import absltest
from absl.testing import parameterized
from tfx.components.trainer.rewriting import rewriter_factory


class RewriterFactoryTest(parameterized.TestCase):

  @parameterized.named_parameters(('TFJS', rewriter_factory.TFJS_REWRITER),
                                  ('TFLite', rewriter_factory.TFLITE_REWRITER))
  def testRewriterFactorySuccessfullyCreatedRewriter(self, rewriter_name):
    tfrw = rewriter_factory.create_rewriter(rewriter_name, name='my_rewriter')
    self.assertTrue(tfrw)
    self.assertEqual(tfrw.name, 'my_rewriter')


if __name__ == '__main__':
  absltest.main()
