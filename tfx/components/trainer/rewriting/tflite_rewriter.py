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
"""Rewriter that invokes the TFLite converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from typing import Text

import six
import tensorflow as tf

from tfx.components.trainer.rewriting import rewriter
from tfx.utils import io_utils

EXTRA_ASSETS_DIRECTORY = 'assets.extra'


def _create_tflite_converter(
    saved_model_path: Text, enable_experimental_new_converter: bool,
    enable_quantization: bool) -> tf.lite.TFLiteConverter:
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.experimental_new_converter = enable_experimental_new_converter
  if enable_quantization:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
  return converter


def _create_tflite_compatible_saved_model(src: Text, dst: Text):
  io_utils.copy_dir(src, dst)
  assets_path = os.path.join(dst, tf.saved_model.ASSETS_DIRECTORY)
  if tf.io.gfile.exists(assets_path):
    tf.io.gfile.rmtree(assets_path)
  assets_extra_path = os.path.join(dst, EXTRA_ASSETS_DIRECTORY)
  if tf.io.gfile.exists(assets_extra_path):
    tf.io.gfile.rmtree(assets_extra_path)


class TFLiteRewriter(rewriter.BaseRewriter):
  """Performs TFLite conversion."""

  def __init__(self,
               name: Text,
               filename: Text = 'tflite',
               enable_experimental_new_converter: bool = False,
               copy_assets: bool = True,
               copy_assets_extra: bool = True,
               enable_quantization: bool = False):
    """Create an instance of the TFLiteRewriter.

    Args:
      name: The name to use when identifying the rewriter.
      filename: The name of the file to use for the tflite model.
      enable_experimental_new_converter: Whether to use the MLIR converter.
      copy_assets: Boolean whether to copy the assets directory to the rewritten
        model directory.
      copy_assets_extra: Boolean whether to copy the assets.extra directory to
        the rewritten model directory.
      enable_quantization: Boolean whether to enable default TFLite
        quantization.
    """
    # TODO(b/152636072): Add support for representative_dataset.
    self._name = name
    self._filename = six.ensure_text(filename)
    self._enable_experimental_new_converter = enable_experimental_new_converter
    self._copy_assets = copy_assets
    self._copy_assets_extra = copy_assets_extra
    self._enable_quantization = enable_quantization

  @property
  def name(self) -> Text:
    """The user-specified name of the rewriter."""
    return self._name

  def _pre_rewrite_validate(self, original_model: rewriter.ModelDescription):
    """Performs pre-rewrite checks to see if the model can be rewritten.

    Args:
      original_model: A `ModelDescription` object describing the model to be
        rewritten.

    Raises:
      ValueError: If the original model does not have the expected structure.
    """
    if original_model.model_type != rewriter.ModelType.SAVED_MODEL:
      raise ValueError('TFLiteRewriter can only convert SavedModels.')

  def _rewrite(self, original_model: rewriter.ModelDescription,
               rewritten_model: rewriter.ModelDescription):
    """Rewrites the provided model.

    Args:
      original_model: A `ModelDescription` specifying the original model to be
        rewritten.
      rewritten_model: A `ModelDescription` specifying the format and location
        of the rewritten model.

    Raises:
      ValueError: If the model could not be sucessfully rewritten.
    """
    if rewritten_model.model_type not in [
        rewriter.ModelType.TFLITE_MODEL, rewriter.ModelType.ANY_MODEL
    ]:
      raise ValueError('TFLiteConverter can only convert to the TFLite format.')

    # TODO(dzats): We create a temporary directory with a SavedModel that does
    # not contain an assets or assets.extra directory. Remove this when the
    # TFLite converter can convert models having these directories.
    tmp_model_dir = os.path.join(
        six.ensure_text(rewritten_model.path),
        'tmp-rewrite-' + str(int(time.time())))
    if tf.io.gfile.exists(tmp_model_dir):
      raise ValueError('TFLiteConverter is unable to create a unique path '
                       'for the temp rewriting directory.')

    tf.io.gfile.makedirs(tmp_model_dir)
    _create_tflite_compatible_saved_model(
        six.ensure_text(original_model.path), tmp_model_dir)

    converter = _create_tflite_converter(
        saved_model_path=tmp_model_dir,
        enable_experimental_new_converter=self
        ._enable_experimental_new_converter,
        enable_quantization=self._enable_quantization)
    tflite_model = converter.convert()

    output_path = os.path.join(
        six.ensure_text(rewritten_model.path), self._filename)
    with tf.io.gfile.GFile(six.ensure_text(output_path), 'wb') as f:
      f.write(six.ensure_binary(tflite_model))
    tf.io.gfile.rmtree(tmp_model_dir)

    copy_pairs = []
    if self._copy_assets:
      src = os.path.join(
          six.ensure_text(original_model.path), tf.saved_model.ASSETS_DIRECTORY)
      dst = os.path.join(
          six.ensure_text(rewritten_model.path),
          tf.saved_model.ASSETS_DIRECTORY)
      if tf.io.gfile.isdir(src):
        tf.io.gfile.mkdir(dst)
        copy_pairs.append((src, dst))
    if self._copy_assets_extra:
      src = os.path.join(
          six.ensure_text(original_model.path), EXTRA_ASSETS_DIRECTORY)
      dst = os.path.join(
          six.ensure_text(rewritten_model.path), EXTRA_ASSETS_DIRECTORY)
      if tf.io.gfile.isdir(src):
        tf.io.gfile.mkdir(dst)
        copy_pairs.append((src, dst))
    for src, dst in copy_pairs:
      io_utils.copy_dir(src, dst)

  def _post_rewrite_validate(self, rewritten_model: rewriter.ModelDescription):
    """Performs post-rewrite checks to see if the rewritten model is valid.

    Args:
      rewritten_model: A `ModelDescription` specifying the format and location
        of the rewritten model.

    Raises:
      ValueError: If the rewritten model is not valid.
    """
    # TODO(dzats): Implement post-rewrite validation.
    pass
