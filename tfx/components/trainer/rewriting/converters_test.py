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
"""Tests for third_party.tfx.components.trainer.rewriting.converters."""

import os
import tempfile

from absl.testing.absltest import mock

import tensorflow as tf

from tfx.components.trainer.rewriting import converters
from tfx.components.trainer.rewriting import rewriter
from tfx.dsl.io import fileio

BASE_EXPORT_SUBDIR = 'export_1'
ORIGINAL_SAVED_MODEL = 'saved_model.pbtxt'
ORIGINAL_VOCAB = 'vocab'
REWRITTEN_SAVED_MODEL = 'rewritten_model.pbtxt'
REWRITTEN_VOCAB = 'rewritten_vocab'


def _export_fn(estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
  del estimator, checkpoint_path, eval_result, is_the_final_export
  path = os.path.join(export_path, BASE_EXPORT_SUBDIR)
  fileio.makedirs(path)
  with fileio.open(os.path.join(path, ORIGINAL_SAVED_MODEL), 'w') as f:
    f.write(str(ORIGINAL_SAVED_MODEL))

  assets_path = os.path.join(path, tf.saved_model.ASSETS_DIRECTORY)
  fileio.makedirs(assets_path)
  with fileio.open(os.path.join(assets_path, ORIGINAL_VOCAB), 'w') as f:
    f.write(str(ORIGINAL_VOCAB))

  return path


class RewritingExporterTest(tf.test.TestCase):

  class _TestRewriter(rewriter.BaseRewriter):

    def __init__(self, rewrite_raises_error):
      """Initializes the MyRewriter class.

      Args:
        rewrite_raises_error: Boolean specifying whether to raise a ValueError.
      """
      self._rewrite_raises_error = rewrite_raises_error
      self.rewrite_called = False

    @property
    def name(self):
      return 'test_rewriter'

    def _pre_rewrite_validate(self, original_model):
      pass

    def _rewrite(self, original_model, rewritten_model):
      self.rewrite_called = True
      assert fileio.exists(
          os.path.join(original_model.path, ORIGINAL_SAVED_MODEL))
      assert fileio.exists(
          os.path.join(original_model.path, tf.saved_model.ASSETS_DIRECTORY,
                       ORIGINAL_VOCAB))
      with fileio.open(
          os.path.join(rewritten_model.path, REWRITTEN_SAVED_MODEL), 'w') as f:
        f.write(str(REWRITTEN_SAVED_MODEL))
      assets_path = os.path.join(rewritten_model.path,
                                 tf.saved_model.ASSETS_DIRECTORY)
      fileio.makedirs(assets_path)
      with fileio.open(os.path.join(assets_path, REWRITTEN_VOCAB), 'w') as f:
        f.write(str(REWRITTEN_VOCAB))
      if self._rewrite_raises_error:
        raise ValueError('rewrite-error')

    def _post_rewrite_validate(self, rewritten_model):
      pass

  def setUp(self):
    super().setUp()
    self._estimator = 'estimator'
    self._export_path = tempfile.mkdtemp()
    self._checkpoint_path = 'checkpoint_path'
    self._eval_result = 'eval_result'
    self._is_the_final_export = True
    self._base_exporter = tf.estimator.FinalExporter(
        name='base_exporter', serving_input_receiver_fn=lambda: None)

  @mock.patch.object(tf.estimator.FinalExporter, 'export')
  def testRewritingExporterSucceeds(self, base_exporter_mock):

    base_exporter_mock.side_effect = _export_fn

    tr = self._TestRewriter(False)
    r_e = converters.RewritingExporter(self._base_exporter, tr)
    final_path = r_e.export(self._estimator, self._export_path,
                            self._checkpoint_path, self._eval_result,
                            self._is_the_final_export)
    self.assertEqual(final_path,
                     os.path.join(self._export_path, BASE_EXPORT_SUBDIR))
    self.assertTrue(
        fileio.exists(os.path.join(final_path, REWRITTEN_SAVED_MODEL)))
    self.assertTrue(
        fileio.exists(
            os.path.join(final_path, tf.saved_model.ASSETS_DIRECTORY,
                         REWRITTEN_VOCAB)))

    base_exporter_mock.assert_called_once_with(self._estimator,
                                               self._export_path,
                                               self._checkpoint_path,
                                               self._eval_result,
                                               self._is_the_final_export)

  @mock.patch.object(tf.estimator.FinalExporter, 'export')
  def testRewritingHandlesNoBaseExport(self, base_exporter_mock):

    base_exporter_mock.return_value = None

    tr = self._TestRewriter(False)
    r_e = converters.RewritingExporter(self._base_exporter, tr)
    final_path = r_e.export(self._estimator, self._export_path,
                            self._checkpoint_path, self._eval_result,
                            self._is_the_final_export)
    self.assertIsNone(final_path)
    self.assertFalse(tr.rewrite_called)

    base_exporter_mock.assert_called_once_with(self._estimator,
                                               self._export_path,
                                               self._checkpoint_path,
                                               self._eval_result,
                                               self._is_the_final_export)

  @mock.patch.object(tf.estimator.FinalExporter, 'export')
  def testRewritingExporterHandlesError(self, base_exporter_mock):

    base_exporter_mock.side_effect = _export_fn

    tr = self._TestRewriter(True)
    r_e = converters.RewritingExporter(self._base_exporter, tr)
    with self.assertRaisesRegex(ValueError, '.*rewrite-error'):
      r_e.export(self._estimator, self._export_path, self._checkpoint_path,
                 self._eval_result, self._is_the_final_export)
    base_exporter_mock.assert_called_once_with(self._estimator,
                                               self._export_path,
                                               self._checkpoint_path,
                                               self._eval_result,
                                               self._is_the_final_export)
    self.assertTrue(tr.rewrite_called)


class RewriteSavedModelTest(tf.test.TestCase):

  @mock.patch.object(converters, '_invoke_rewriter')
  def testRewritingExporterSucceeds(self, invoke_rewriter_mock):
    src = '/my/src'
    dst = '/my/dst'
    rewriter_inst = 'r1'
    converters.rewrite_saved_model(src, dst, rewriter_inst)
    invoke_rewriter_mock.assert_called_once_with(src, dst, rewriter_inst,
                                                 rewriter.ModelType.SAVED_MODEL,
                                                 rewriter.ModelType.SAVED_MODEL)


if __name__ == '__main__':
  tf.test.main()
