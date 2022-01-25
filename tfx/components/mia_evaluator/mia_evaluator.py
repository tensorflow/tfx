# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tensorflow Model Analysis MIA Evaluator."""
import logging
import random
from typing import Any, Dict, Iterable, Tuple, Optional

import apache_beam as beam
import numpy as np
import tensorflow_model_analysis as tfma
import tensorflow_model_analysis.utils.util as tfma_utils

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import data_structures as mia_ds
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia

from tfx.components import mia_evaluator

# Sample size for the size of the train and test splits. Each split will have
# roughly this many elements.
SAMPLE_SIZE = 10000
TRAIN_SPLIT_NAME = 'train'
TEST_SPLIT_NAME = 'test'
SPLIT_KEY = 'split'
FILTER_KEY_DELIMITER = ','


def MiaEvaluator(
    extractors: Iterable[tfma.extractors.Extractor],
    mia_config: mia_evaluator.MIACustomConfig,
    eval_config: tfma.EvalConfig,
    metrics_key: str = tfma.constants.METRICS_KEY
) -> tfma.evaluators.evaluator.Evaluator:
  """Creates a MIA Evaluator."""
  # pylint: disable=no-value-for-parameter
  return tfma.evaluators.evaluator.Evaluator(
      stage_name='EvaluateMIA',
      run_after=None,  # Runs first so train examples won't be filtered out.
      ptransform=_EvaluateMIAPTransform(
          metrics_key=metrics_key,
          mia_config=mia_config,
          extractors=extractors,
          eval_config=eval_config))


def _OnlyKeepLabelsAndPredictions(example: Dict[str, Any]) -> Dict[str, Any]:
  """Returns a much smaller dict only containing labels and predictions."""
  result = dict()
  result[tfma.constants.LABELS_KEY] = example[tfma.constants.LABELS_KEY]
  result[tfma.constants.PREDICTIONS_KEY] = example[
      tfma.constants.PREDICTIONS_KEY]
  return result


def FilterExtracts(extracts: beam.pvalue.PCollection,
                   split_type: str) -> beam.pvalue.PCollection:
  """Samples extracts uniformly at random."""
  extracts_count = (
      extracts | f'Count[{split_type}]' >> beam.combiners.Count.Globally())
  return (extracts | f'Sample[{split_type}]' >> beam.Filter(
      lambda x, count: random.random() < float(SAMPLE_SIZE) / count,
      count=beam.pvalue.AsSingleton(extracts_count))
          | f'OnlyKeepLabelsAndPredictions[{split_type}]' >>
          beam.Map(_OnlyKeepLabelsAndPredictions))


def PartitionSplits(x: Any, unused_num_partitions: int) -> int:
  """Partitions splits into 3 categories, depending on the split_name."""
  split = x[SPLIT_KEY][0].split_name
  if split == TRAIN_SPLIT_NAME:
    return 0
  if split == TEST_SPLIT_NAME:
    return 1
  return 2


@beam.ptransform_fn
@beam.typehints.with_input_types(tfma.types.Extracts)
@beam.typehints.with_output_types(Any)
def _EvaluateMIAPTransform(
    extracts: beam.pvalue.PCollection,
    extractors: Iterable[tfma.extractors.Extractor],
    mia_config: mia_evaluator.MIACustomConfig, eval_config: tfma.EvalConfig,
    metrics_key: str) -> tfma.evaluators.evaluator.Evaluation:
  """Evaluates MIA."""
  train_extracts, test_extracts, unused_extracts = (
      extracts | 'Partition' >> beam.Partition(PartitionSplits, 3))

  for x in extractors:
    train_extracts = (
        train_extracts | x.stage_name + TRAIN_SPLIT_NAME >> x.ptransform)
    test_extracts = (
        test_extracts | x.stage_name + TEST_SPLIT_NAME >> x.ptransform)

  filtered_train_extracts = FilterExtracts(train_extracts, TRAIN_SPLIT_NAME)
  filtered_test_extracts = FilterExtracts(test_extracts, TEST_SPLIT_NAME)

  # Extract model names from EvalConfig
  model_names = [
      spec.name if hasattr(spec, 'name') else None
      for spec in eval_config.model_specs
  ]

  mia_output = (
      extracts.pipeline
      | 'CreateSingletonList' >> beam.Create([None])
      | 'RunMia' >> beam.Map(
          _RunMiaInMemory,
          train_extracts=beam.pvalue.AsList(filtered_train_extracts),
          test_extracts=beam.pvalue.AsList(filtered_test_extracts),
          mia_config=mia_config,
          model_names=model_names))
  # Add the default TFMA slice to the output for correct post-processing.
  results = (mia_output | 'AddSlice' >> beam.Map(lambda x: ((), x)))
  return {metrics_key: results}


def _RunMiaInMemory(
    unused_extracts: Any,
    train_extracts: Iterable[Any],
    test_extracts: Iterable[Any],
    mia_config: mia_evaluator.MIACustomConfig,
    model_names: Iterable[Any] = (None,)
) -> Dict[tfma.metrics.MetricKey, float]:
  """Runs MIA on two in-memory iterables.

    Since each input only contains 10k examples, both should fit in memory.
  Args:
    unused_extracts: Unused placeholder pcol
    train_extracts: Train extracts
    test_extracts: Test extracts
    mia_config: Mia custom config.
    model_names: A list of model names. Computes one set of metrics per model.

  Returns:
    MIA results.
  """
  mia_metrics = dict()
  for model_name in model_names:
    probabilities_train, labels_train = _ExtractProbabilitiesAndLabels(
        train_extracts, mia_config, model_name)
    probabilities_test, labels_test = _ExtractProbabilitiesAndLabels(
        test_extracts, mia_config, model_name)
    # MIA library expects predictions to be in the shape
    # (num_samples, num_classes). The taxi pipeline produces predictions in
    # the shape (num_samples, 1) since it's a binary classifier.
    labels_train = None
    labels_test = None
    attack_input_data = mia_ds.AttackInputData(
        probs_train=probabilities_train,
        probs_test=probabilities_test,
        labels_train=labels_train,
        labels_test=labels_test)

    attack_types = [
        mia_ds.AttackType.LOGISTIC_REGRESSION, mia_ds.AttackType.RANDOM_FOREST,
        mia_ds.AttackType.K_NEAREST_NEIGHBORS,
        mia_ds.AttackType.MULTI_LAYERED_PERCEPTRON
    ]

    if labels_train is not None:
      attack_types.append(mia_ds.AttackType.THRESHOLD_ATTACK)

    attack_results = mia.run_attacks(
        attack_input_data,
        mia_ds.SlicingSpec(entire_dataset=True),
        attack_types=attack_types)
    mia_metrics[tfma.metrics.MetricKey(
        name='mia_auc_score', model_name=model_name
    )] = attack_results.get_result_with_max_auc().get_auc()
    mia_metrics[tfma.metrics.MetricKey(
        name='mia_attacker_advantage_score', model_name=model_name
    )] = attack_results.get_result_with_max_attacker_advantage(
    ).get_attacker_advantage()
  return mia_metrics


def _FlattenArrayOfArrays(array_of_arrays: np.ndarray) -> np.ndarray:
  """Flattens array of arrays."""
  if isinstance(array_of_arrays, np.ndarray) and isinstance(
      array_of_arrays[0], np.ndarray):
    if len(array_of_arrays) != 1:
      raise ValueError(
          f'Array of arrays {array_of_arrays} contains more than 1 subarray.')
    return array_of_arrays[0]
  return array_of_arrays


def _ExtractProbabilitiesAndLabels(
    extracts: Iterable[Any], mia_config: mia_evaluator.MIACustomConfig,
    model_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
  """Returns predictions and labels from extractor output."""
  probabilities = None
  labels = None
  for example in extracts:
    standard_extracts = tfma_utils.StandardExtracts(example)
    example_probabilities = np.round(
        _FlattenArrayOfArrays(
            standard_extracts.get_predictions(
                model_name=model_name, output_name=mia_config.output_name)), 3)
    example_label = _FlattenArrayOfArrays(
        standard_extracts.get_labels(
            model_name=model_name, output_name=mia_config.output_name))

    if probabilities is None:
      probabilities = example_probabilities
      labels = example_label
    else:
      probabilities = np.vstack((probabilities, example_probabilities))
      if example_label is not None:
        labels = np.append(
            np.asarray(labels, dtype=np.int),
            np.asarray(example_label, dtype=np.int),
            axis=0)
  if probabilities is None:
    raise ValueError('Unable to extract probabilities.')
  # Some models don't provide a single label per example. If so, only return
  # probabilities.
  if len(probabilities) != len(labels):
    logging.warning(
        'Extracted %s probabilities, but %s labels. Only using probabilities for the test.',
        len(probabilities), len(labels))
    labels = None
  return probabilities, labels
