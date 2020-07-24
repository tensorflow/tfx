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
"""Tests for third_party.tfx.components.trainer.rewriting.tfjs_rewriter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock

import tensorflow as tf

from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import tfjs_rewriter


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
