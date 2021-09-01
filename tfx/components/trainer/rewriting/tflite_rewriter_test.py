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

import os
import tempfile

from unittest import mock
import numpy as np

import tensorflow as tf

from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import tflite_rewriter
from tfx.dsl.io import fileio

EXTRA_ASSETS_DIRECTORY = 'assets.extra'


class TFLiteRewriterTest(tf.test.TestCase):

  class ConverterMock:

    class TargetSpec:
      pass

    target_spec = TargetSpec()

    def convert(self):
      return 'model'

  def create_temp_model_template(self):
    src_model_path = tempfile.mkdtemp()
    dst_model_path = tempfile.mkdtemp()

    saved_model_path = os.path.join(src_model_path,
                                    tf.saved_model.SAVED_MODEL_FILENAME_PBTXT)
    with fileio.open(saved_model_path, 'wb') as f:
      f.write(b'saved_model')

    src_model = rewriter.ModelDescription(rewriter.ModelType.SAVED_MODEL,
                                          src_model_path)
    dst_model = rewriter.ModelDescription(rewriter.ModelType.TFLITE_MODEL,
                                          dst_model_path)

    return src_model, dst_model, src_model_path, dst_model_path

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterNoAssetsSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, _, dst_model_path = self.create_temp_model_template()

    tfrw = tflite_rewriter.TFLiteRewriter(name='myrw', filename='fname')
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[],
        quantization_supported_types=[],
        representative_dataset=None,
        signature_key=None)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(fileio.exists(expected_model))
    with fileio.open(expected_model, 'rb') as f:
      self.assertEqual(f.read(), b'model')

  @mock.patch('tfx.components.trainer.rewriting'
              '.tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterWithAssetsSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, src_model_path, dst_model_path = (
        self.create_temp_model_template())

    assets_dir = os.path.join(src_model_path, tf.saved_model.ASSETS_DIRECTORY)
    fileio.mkdir(assets_dir)
    assets_file_path = os.path.join(assets_dir, 'assets_file')
    with fileio.open(assets_file_path, 'wb') as f:
      f.write(b'assets_file')

    assets_extra_dir = os.path.join(src_model_path, EXTRA_ASSETS_DIRECTORY)
    fileio.mkdir(assets_extra_dir)
    assets_extra_file_path = os.path.join(assets_extra_dir, 'assets_extra_file')
    with fileio.open(assets_extra_file_path, 'wb') as f:
      f.write(b'assets_extra_file')

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        quantization_optimizations=[tf.lite.Optimize.DEFAULT])
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_supported_types=[],
        representative_dataset=None,
        signature_key=None)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(fileio.exists(expected_model))
    with fileio.open(expected_model, 'rb') as f:
      self.assertEqual(f.read(), b'model')

    expected_assets_file = os.path.join(dst_model_path,
                                        tf.saved_model.ASSETS_DIRECTORY,
                                        'assets_file')
    with fileio.open(expected_assets_file, 'rb') as f:
      self.assertEqual(f.read(), b'assets_file')

    expected_assets_extra_file = os.path.join(dst_model_path,
                                              EXTRA_ASSETS_DIRECTORY,
                                              'assets_extra_file')
    with fileio.open(expected_assets_extra_file, 'rb') as f:
      self.assertEqual(f.read(), b'assets_extra_file')

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterQuantizationHybridSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, _, dst_model_path = self.create_temp_model_template()

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        quantization_optimizations=[tf.lite.Optimize.DEFAULT])
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_supported_types=[],
        representative_dataset=None,
        signature_key=None)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(fileio.exists(expected_model))
    with fileio.open(expected_model, 'rb') as f:
      self.assertEqual(f.read(), b'model')

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterQuantizationFloat16Succeeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, _, dst_model_path = self.create_temp_model_template()

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_supported_types=[tf.float16])
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_supported_types=[tf.float16],
        representative_dataset=None,
        signature_key=None)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(fileio.exists(expected_model))
    with fileio.open(expected_model, 'rb') as f:
      self.assertEqual(f.read(), b'model')

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter._create_tflite_compatible_saved_model')
  @mock.patch('tensorflow.lite.TFLiteConverter.from_saved_model')
  def testInvokeTFLiteRewriterQuantizationFullIntegerFailsNoData(
      self, converter, model):

    class ModelMock:
      pass

    m = ModelMock()
    model.return_value = m
    n = self.ConverterMock()
    converter.return_value = n

    with self.assertRaises(ValueError):
      _ = tflite_rewriter.TFLiteRewriter(
          name='myrw',
          filename='fname',
          quantization_optimizations=[tf.lite.Optimize.DEFAULT],
          quantization_enable_full_integer=True)

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeTFLiteRewriterQuantizationFullIntegerSucceeds(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, _, dst_model_path = self.create_temp_model_template()

    def representative_dataset():
      for i in range(2):
        yield [np.array(i)]

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_enable_full_integer=True,
        representative_dataset=representative_dataset)
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[tf.lite.Optimize.DEFAULT],
        quantization_supported_types=[],
        representative_dataset=representative_dataset,
        signature_key=None)
    expected_model = os.path.join(dst_model_path, 'fname')
    self.assertTrue(fileio.exists(expected_model))
    with fileio.open(expected_model, 'rb') as f:
      self.assertEqual(f.read(), b'model')

  @mock.patch('tensorflow.lite.TFLiteConverter.from_saved_model')
  def testInvokeTFLiteRewriterWithSignatureKey(self, converter):
    m = self.ConverterMock()
    converter.return_value = m

    src_model, dst_model, _, _ = self.create_temp_model_template()

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw',
        filename='fname',
        signature_key='tflite')
    tfrw.perform_rewrite(src_model, dst_model)

    _, kwargs = converter.call_args
    self.assertListEqual(kwargs['signature_keys'], ['tflite'])

  @mock.patch('tfx.components.trainer.rewriting.'
              'tflite_rewriter.TFLiteRewriter._create_tflite_converter')
  def testInvokeConverterWithKwargs(self, converter):
    converter.return_value = self.ConverterMock()

    src_model, dst_model, _, _ = self.create_temp_model_template()

    tfrw = tflite_rewriter.TFLiteRewriter(
        name='myrw', filename='fname', output_arrays=['head'])
    tfrw.perform_rewrite(src_model, dst_model)

    converter.assert_called_once_with(
        saved_model_path=mock.ANY,
        quantization_optimizations=[],
        quantization_supported_types=[],
        representative_dataset=None,
        signature_key=None,
        output_arrays=['head'])


if __name__ == '__main__':
  tf.test.main()
