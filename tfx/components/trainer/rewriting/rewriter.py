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
"""Base class that TFX rewriters inherit and invocation utilities."""

import abc
import collections
import enum


ModelDescription = collections.namedtuple('ModelDescription',
                                          ['model_type', 'path'])


class ModelType(enum.Enum):
  """Types of models used or created by the rewriter."""
  ANY_MODEL = 1
  SAVED_MODEL = 2
  TFLITE_MODEL = 3
  TFJS_MODEL = 4


class BaseRewriter(abc.ABC):
  """Base class from which all rewriters should inherit."""

  @abc.abstractproperty
  def name(self) -> str:
    """Name of the rewriter.

    Should not be `None` nor empty.
    """
    pass

  @abc.abstractmethod
  def _pre_rewrite_validate(self, original_model: ModelDescription):
    """Perform pre-rewrite validation to check the model has expected structure.

    Args:
      original_model: A `ModelDescription` object describing the original model.

    Raises:
      ValueError: If the original model does not have the expected structure.
    """
    pass

  @abc.abstractmethod
  def _rewrite(self, original_model: ModelDescription,
               rewritten_model: ModelDescription):
    """Perform the rewrite.

    Args:
      original_model: A `ModelDescription` object describing the original model.
      rewritten_model: A `ModelDescription` object describing the location and
        type of the rewritten output.

    Raises:
      ValueError: If the original model was not successfully rewritten.
    """
    pass

  @abc.abstractmethod
  def _post_rewrite_validate(self, rewritten_model: ModelDescription):
    """Perform post-rewrite validation.

    Args:
      rewritten_model: A `ModelDescription` object describing the location and
        type of the rewritten output.

    Raises:
      ValueError: If the rewritten model is not valid.
    """
    pass

  def perform_rewrite(self, original_model: ModelDescription,
                      rewritten_model: ModelDescription):
    """Invoke all validations and perform the rewrite.

    Args:
      original_model: A `base_rewriter.ModelDescription` object describing the
        original model.
      rewritten_model: A `base_rewriter.ModelDescription` object describing the
        location and type of the rewritten model.

    Raises:
      ValueError: if the model was not successfully rewritten.
    """
    try:
      self._pre_rewrite_validate(original_model)
    except ValueError as v:
      raise ValueError('{} failed to perform pre-rewrite validation. Original '
                       'model: {}. Error: {}'.format(self.name,
                                                     str(original_model),
                                                     str(v)))

    try:
      self._rewrite(original_model, rewritten_model)
    except ValueError as v:
      raise ValueError(
          '{} failed to rewrite model. Original model: {}. Error {}'.format(
              self.name, str(original_model), str(v)))

    try:
      self._post_rewrite_validate(rewritten_model)
    except ValueError as v:
      raise ValueError(
          '{} failed to validate rewritten model. Rewritten model: {}. Error {}'
          .format(self.name, str(rewritten_model), str(v)))
