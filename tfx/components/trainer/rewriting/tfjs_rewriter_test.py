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
"""Tests for third_party.tfx.components.trainer.rewriting.tfjs_rewriter."""

import unittest
from unittest import mock

import tensorflow as tf

from tfx.components.trainer.rewriting import rewriter


try:
  from tfx.components.trainer.rewriting import tfjs_rewriter  # pylint: disable=g-import-not-at-top
except ImportError as err:
  tfjs_rewriter = None


@unittest.skipIf(tf.__version__ < '2',
                 'TFJS requires TF2 which is not satisfied for TF1 environment,'
                 ' thus skip any TFJS related tests.')
@unittest.skipIf(tfjs_rewriter is None,
                 'Cannot import tfjs_rewriter. This can happen when tfjs is not'
                 ' available.')
class TFJSRewriterTest(tf.test.TestCase):

  @mock.patch('tfx.components.trainer.rewriting.'
              'tfjs_rewriter._convert_tfjs_model')
  def testInvokeTFJSRewriter(self, converter):
    src_model_path = '/path/to/src/model'
    dst_model_path = '/path/to/dst/model'

    src_model = rewriter.ModelDescription(rewriter.ModelType.SAVED_MODEL,
                                          src_model_path)
    dst_model = rewriter.ModelDescription(rewriter.ModelType.TFJS_MODEL,
                                          dst_model_path)

    tfrw = tfjs_rewriter.TFJSRewriter(name='myrw')
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(src_model_path, dst_model_path)


if __name__ == '__main__':
  tf.test.main()
