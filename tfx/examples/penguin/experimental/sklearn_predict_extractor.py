# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Predict extractor for scikit-learn models."""

import copy
import os
import pickle
from typing import Dict, Iterable, List

import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx_bsl.tfxio import tensor_adapter

_PREDICT_EXTRACTOR_STAGE_NAME = 'SklearnPredict'


def _make_sklearn_predict_extractor(
    eval_shared_model: tfma.EvalSharedModel,) -> tfma.extractors.Extractor:
  """Creates an extractor for performing predictions using a scikit-learn model.

  The extractor's PTransform loads and runs the serving pickle against
  every extract yielding a copy of the incoming extracts with an additional
  extract added for the predictions keyed by tfma.PREDICTIONS_KEY. The model
  inputs are searched for under tfma.FEATURES_KEY.

  Args:
    eval_shared_model: Shared model (single-model evaluation).

  Returns:
    Extractor for extracting predictions.
  """
  eval_shared_models = tfma.utils.verify_and_update_eval_shared_models(
      eval_shared_model)
  return tfma.extractors.Extractor(
      stage_name=_PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractPredictions(  # pylint: disable=no-value-for-parameter
          eval_shared_models={m.model_name: m for m in eval_shared_models}))


@beam.typehints.with_input_types(tfma.Extracts)
@beam.typehints.with_output_types(tfma.Extracts)
class _TFMAPredictionDoFn(tfma.utils.DoFnWithModels):
  """A DoFn that loads the models and predicts."""

  def __init__(self, eval_shared_models: Dict[str, tfma.EvalSharedModel]):
    super().__init__({k: v.model_loader for k, v in eval_shared_models.items()})

  def setup(self):
    super().setup()
    self._feature_keys = None
    self._label_key = None
    for loaded_model in self._loaded_models.values():
      if self._feature_keys and self._label_key:
        assert self._feature_keys == loaded_model.feature_keys, (
            f'Features mismatch in loaded models. Expected {self._feature_keys}'
            f', got {loaded_model.feature_keys} instead.')
        assert self._label_key == loaded_model.label_key, (
            f'Label mismatch in loaded models. Expected "{self._label_key}"'
            f', got "{loaded_model.label_key}" instead.')
      elif loaded_model.feature_keys and loaded_model.label_key:
        self._feature_keys = loaded_model.feature_keys
        self._label_key = loaded_model.label_key
      else:
        raise ValueError('Missing feature or label keys in loaded model.')

  def process(self, elem: tfma.Extracts) -> Iterable[tfma.Extracts]:
    """Uses loaded models to make predictions on batches of data.

    Args:
      elem: An extract containing batched features.

    Yields:
      Copy of the original extracts with predictions added for each model. If
      there are multiple models, a list of dicts keyed on model names will be
      added, with each value corresponding to a prediction for a single sample.
    """
    # Build feature and label vectors because sklearn cannot read tf.Examples.
    features = []
    labels = []
    result = copy.copy(elem)
    for features_dict in result[tfma.FEATURES_KEY]:
      features_row = [features_dict[key] for key in self._feature_keys]
      features.append(np.concatenate(features_row))
      labels.append(features_dict[self._label_key])
    result[tfma.LABELS_KEY] = np.concatenate(labels)

    # Generate predictions for each model.
    for model_name, loaded_model in self._loaded_models.items():
      preds = loaded_model.predict(features)
      if len(self._loaded_models) == 1:
        result[tfma.PREDICTIONS_KEY] = preds
      elif tfma.PREDICTIONS_KEY not in result:
        result[tfma.PREDICTIONS_KEY] = [{model_name: pred} for pred in preds]
      else:
        for i, pred in enumerate(preds):
          result[tfma.PREDICTIONS_KEY][i][model_name] = pred
    yield result


@beam.ptransform_fn
@beam.typehints.with_input_types(tfma.Extracts)
@beam.typehints.with_output_types(tfma.Extracts)
def _ExtractPredictions(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_models: Dict[str, tfma.EvalSharedModel],
) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions and possibly other tensors to extracts.

  Args:
    extracts: PCollection of extracts with inputs keyed by tfma.INPUTS_KEY.
    eval_shared_models: Shared model parameters keyed by model name.

  Returns:
    PCollection of Extracts updated with the predictions.
  """
  return extracts | 'Predict' >> beam.ParDo(
      _TFMAPredictionDoFn(eval_shared_models))


def _custom_model_loader_fn(model_path: str):
  """Returns a function that loads a scikit-learn model."""
  return lambda: pickle.load(tf.io.gfile.GFile(model_path, 'rb'))


# TFX Evaluator will call the following functions.
def custom_eval_shared_model(
    eval_saved_model_path, model_name, eval_config,
    **kwargs) -> tfma.EvalSharedModel:
  """Returns a single custom EvalSharedModel."""
  model_path = os.path.join(eval_saved_model_path, 'model.pkl')
  return tfma.default_eval_shared_model(
      eval_saved_model_path=model_path,
      model_name=model_name,
      eval_config=eval_config,
      custom_model_loader=tfma.ModelLoader(
          construct_fn=_custom_model_loader_fn(model_path)),
      add_metrics_callbacks=kwargs.get('add_metrics_callbacks'))


def custom_extractors(
    eval_shared_model: tfma.MaybeMultipleEvalSharedModels,
    eval_config: tfma.EvalConfig,
    tensor_adapter_config: tensor_adapter.TensorAdapterConfig,
) -> List[tfma.extractors.Extractor]:
  """Returns default extractors plus a custom prediction extractor."""
  predict_extractor = _make_sklearn_predict_extractor(eval_shared_model)
  return tfma.default_extractors(
      eval_shared_model=eval_shared_model,
      eval_config=eval_config,
      tensor_adapter_config=tensor_adapter_config,
      custom_predict_extractor=predict_extractor)
