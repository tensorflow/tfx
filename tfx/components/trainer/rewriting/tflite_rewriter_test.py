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
"""Tests for third_party.tfx.components.trainer.rewriting.tflite_rewriter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import mock
import six

import tensorflow as tf

from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import tflite_rewriter

EXTRA_ASSETS_DIRECTORY = 'assets.extra'


class TFLiteRewriterTest(tf.test.TestCase):

  class ConverterMock(object):

    def convert(self):
      return 'model'

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterNoAssetsSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model_path = tempfile.mkdtemp()
    dst_model_path = tempfile.mkdtemp()

    saved_model_path = os.path.join(src_model_path,
                                    tf.saved_model.SAVED_MODEL_FILENAME_PBTXT)
    with tf.io.gfile.GFile(saved_model_path, 'wb') as f:
      f.write(six.ensure_binary('saved_model'))

    src_model = rewriter.ModelDescription(rewriter.ModelType.SAVED_MODEL,
                                          src_model_path)
    dst_model = rewriter.ModelDescription(rewriter.ModelType.TFLITE_MODEL,
                                          dst_model_path)

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        enable_experimental_new_converter=True)
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        enable_experimental_new_converter=True,
        enable_quantization=False)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(tf.io.gfile.exists(expected_model))
    with tf.io.gfile.GFile(expected_model, 'rb') as f:
      self.assertEqual(six.ensure_text(f.readline()), 'model')

  @mock.patch('tfx.components.trainer.rewriting'
              '.tflite_rewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterWithAssetsSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model_path = tempfile.mkdtemp()
    dst_model_path = tempfile.mkdtemp()

    saved_model_path = os.path.join(src_model_path,
                                    tf.saved_model.SAVED_MODEL_FILENAME_PBTXT)
    with tf.io.gfile.GFile(saved_model_path, 'wb') as f:
      f.write(six.ensure_binary('saved_model'))

    assets_dir = os.path.join(src_model_path, tf.saved_model.ASSETS_DIRECTORY)
    tf.io.gfile.mkdir(assets_dir)
    assets_file_path = os.path.join(assets_dir, 'assets_file')
    with tf.io.gfile.GFile(assets_file_path, 'wb') as f:
      f.write(six.ensure_binary('assets_file'))

    assets_extra_dir = os.path.join(src_model_path, EXTRA_ASSETS_DIRECTORY)
    tf.io.gfile.mkdir(assets_extra_dir)
    assets_extra_file_path = os.path.join(assets_extra_dir, 'assets_extra_file')
    with tf.io.gfile.GFile(assets_extra_file_path, 'wb') as f:
      f.write(six.ensure_binary('assets_extra_file'))

    src_model = rewriter.ModelDescription(rewriter.ModelType.SAVED_MODEL,
                                          src_model_path)
    dst_model = rewriter.ModelDescription(rewriter.ModelType.TFLITE_MODEL,
                                          dst_model_path)

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        enable_experimental_new_converter=True,
        enable_quantization=True)
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        enable_experimental_new_converter=True,
        enable_quantization=True)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(tf.io.gfile.exists(expected_model))
    with tf.io.gfile.GFile(expected_model, 'rb') as f:
      self.assertEqual(six.ensure_text(f.readline()), 'model')

    expected_assets_file = os.path.join(dst_model_path,
                                        tf.saved_model.ASSETS_DIRECTORY,
                                        'assets_file')
    with tf.io.gfile.GFile(expected_assets_file, 'rb') as f:
      self.assertEqual(six.ensure_text(f.readline()), 'assets_file')

    expected_assets_extra_file = os.path.join(dst_model_path,
                                              EXTRA_ASSETS_DIRECTORY,
                                              'assets_extra_file')
    with tf.io.gfile.GFile(expected_assets_extra_file, 'rb') as f:
      self.assertEqual(six.ensure_text(f.readline()), 'assets_extra_file')


if __name__ == '__main__':
  tf.test.main()
