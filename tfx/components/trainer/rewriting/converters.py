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
"""Converters rewrite models using the provided rewriters."""

import os
import time


import tensorflow as tf
from tfx.components.trainer.rewriting import rewriter
from tfx.dsl.io import fileio


def _invoke_rewriter(src: str, dst: str, rewriter_inst: rewriter.BaseRewriter,
                     src_model_type: rewriter.ModelType,
                     dst_model_type: rewriter.ModelType):
  """Converts the provided model by invoking the specified rewriters.

  Args:
    src: Path to the source model.
    dst: Path where the destination model is to be written.
    rewriter_inst: instance of the rewriter to invoke.
    src_model_type: the `rewriter.ModelType` of the source model.
    dst_model_type: the `rewriter.ModelType` of the destination model.

  Raises:
    ValueError: if the source path is the same as the destination path.
  """

  if src == dst:
    raise ValueError('Source path and destination path cannot match.')

  original_model = rewriter.ModelDescription(src_model_type, src)
  rewritten_model = rewriter.ModelDescription(dst_model_type, dst)

  rewriter_inst.perform_rewrite(original_model, rewritten_model)


class RewritingExporter(tf.estimator.Exporter):
  """This class invokes the base exporter and a series of rewriters."""

  def __init__(self, base_exporter: tf.estimator.Exporter,
               rewriter_inst: rewriter.BaseRewriter):
    """Initializes the rewriting exporter.

    Args:
      base_exporter: The exporter of the original model.
      rewriter_inst: The rewriter instance to invoke. Must inherit from
        `rewriter.BaseRewriter`.
    """
    self._base_exporter = base_exporter
    self._rewriter_inst = rewriter_inst

  @property
  def name(self):
    """Name of the exporter."""
    return self._base_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    """Exports the given `Estimator` to a specific format.

    Performs the export as defined by the base_exporter and invokes all of the
    specified rewriters.

    Args:
      estimator: the `Estimator` to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of `Estimator.evaluate` on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training.  It is False for the intermediate exports during the
        training. When passing `Exporter` to `tf.estimator.train_and_evaluate`
        `is_the_final_export` is always False if `TrainSpec.max_steps` is
        `None`.

    Returns:
      The string path to the base exported directory or `None` if export is
        skipped.

    Raises:
      RuntimeError: Unable to create a temporary rewrite directory.
    """
    base_path = self._base_exporter.export(estimator, export_path,
                                           checkpoint_path, eval_result,
                                           is_the_final_export)
    if not base_path:
      return None

    tmp_rewrite_folder = 'tmp-rewrite-' + str(int(time.time()))
    tmp_rewrite_path = os.path.join(export_path, tmp_rewrite_folder)
    if fileio.exists(tmp_rewrite_path):
      raise RuntimeError('Unable to create a unique temporary rewrite path.')
    fileio.makedirs(tmp_rewrite_path)

    _invoke_rewriter(base_path, tmp_rewrite_path, self._rewriter_inst,
                     rewriter.ModelType.SAVED_MODEL,
                     rewriter.ModelType.ANY_MODEL)

    fileio.rmtree(base_path)
    fileio.rename(tmp_rewrite_path, base_path)
    return base_path


def rewrite_saved_model(
    src: str,
    dst: str,
    rewriter_inst: rewriter.BaseRewriter,
    dst_model_type: rewriter.ModelType = rewriter.ModelType.SAVED_MODEL):
  """Rewrites the provided SavedModel.

  Args:
    src: location of the saved_model to rewrite.
    dst: location of the rewritten saved_model.
    rewriter_inst: the rewriter instance to invoke. Must inherit from
      `rewriter.BaseRewriter`.
    dst_model_type: the `rewriter.ModelType` of the destination model.
  """
  _invoke_rewriter(src, dst, rewriter_inst, rewriter.ModelType.SAVED_MODEL,
                   dst_model_type)
