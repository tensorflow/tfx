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
"""Rewriter that invokes the TFJS converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import six

from tensorflowjs.converters import converter

from tfx.components.trainer.rewriting import rewriter

CONVERTER_SAVED_MODEL_INPUT_FLAG = '--input_format=tf_saved_model'
CONVERTER_SERVING_TAG_FLAG = '--saved_model_tags=serve'
CONVERTER_DEFAULT_SIGNATURE_FLAG = '--signature_name=serving_default'


def _convert_tfjs_model(saved_model_path: Text, destination_path: Text):
  converter.convert([
      CONVERTER_SAVED_MODEL_INPUT_FLAG, CONVERTER_SERVING_TAG_FLAG,
      CONVERTER_DEFAULT_SIGNATURE_FLAG,
      saved_model_path, destination_path
  ])


class TFJSRewriter(rewriter.BaseRewriter):
  """Performs TFJS conversion."""

  def __init__(self, name: Text):
    """Create an instance of the TFJSRewriter.

    Args:
      name: The name to use when identifying the rewriter.
    """
    self._name = name

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
      raise ValueError('TFJSRewriter can only convert SavedModels.')

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
        rewriter.ModelType.TFJS_MODEL, rewriter.ModelType.ANY_MODEL
    ]:
      raise ValueError('TFJSConverter can only convert to the TFJS format.')

    _convert_tfjs_model(
        six.ensure_text(original_model.path),
        six.ensure_text(rewritten_model.path))

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
