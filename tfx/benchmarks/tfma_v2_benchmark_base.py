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
"""TFMA v2 benchmark."""

import copy
import time


import apache_beam as beam
import numpy as np
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.evaluators import poisson_bootstrap
from tensorflow_model_analysis.extractors import example_weights_extractor
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import labels_extractor
from tensorflow_model_analysis.extractors import legacy_input_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.extractors import unbatch_extractor
from tensorflow_model_analysis.metrics import metric_specs as metric_specs_util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.utils import model_util
from tensorflow_model_analysis.utils import util
import tfx
from tfx.benchmarks import benchmark_utils
from tfx.benchmarks import benchmark_base
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import test_util

# Maximum number of examples within a record batch.
_BATCH_SIZE = 1000

# Number of iterations.
_ITERS = 1


# TODO(b/147827582): Also add "TF-level" Keras benchmarks for how TFMAv2
# gets predictions / computes metrics.
class TFMAV2BenchmarkBase(benchmark_base.BenchmarkBase):
  """TFMA benchmark."""

  def __init__(self, dataset, **kwargs):
    # Benchmark runners may pass extraneous arguments we don't care about.
    del kwargs
    super().__init__()
    self._dataset = dataset

  def _init_model(self, multi_model, validation):
    # The benchmark runner will instantiate this class twice - once to determine
    # the benchmarks to run, and once to actually to run them. However, Keras
    # freezes if we try to load the same model twice. As such, we have to pull
    # the model loading out of the constructor into a separate method which we
    # call before each benchmark.
    if multi_model:
      metric_specs = metric_specs_util.specs_from_metrics(
          [tf.keras.metrics.AUC(name="auc", num_thresholds=10000)],
          model_names=["candidate", "baseline"])
      if validation:
        # Only one metric, adding a threshold for all slices.
        metric_specs[0].metrics[0].threshold.CopyFrom(
            tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(
                    lower_bound={"value": 0.5}, upper_bound={"value": 0.5}),
                change_threshold=tfma.GenericChangeThreshold(
                    absolute={"value": -0.001},
                    direction=tfma.MetricDirection.HIGHER_IS_BETTER)))
      self._eval_config = tfma.EvalConfig(
          model_specs=[
              tfma.ModelSpec(name="candidate", label_key="tips"),
              tfma.ModelSpec(
                  name="baseline", label_key="tips", is_baseline=True)
          ],
          metrics_specs=metric_specs)
      self._eval_shared_models = {
          "candidate":
              tfma.default_eval_shared_model(
                  self._dataset.trained_saved_model_path(),
                  eval_config=self._eval_config,
                  model_name="candidate"),
          "baseline":
              tfma.default_eval_shared_model(
                  self._dataset.trained_saved_model_path(),
                  eval_config=self._eval_config,
                  model_name="baseline")
      }
    else:
      metric_specs = metric_specs_util.specs_from_metrics(
          [tf.keras.metrics.AUC(name="auc", num_thresholds=10000)])
      if validation:
        # Only one metric, adding a threshold for all slices.
        metric_specs[0].metrics[0].threshold.CopyFrom(
            tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(
                    lower_bound={"value": 0.5}, upper_bound={"value": 0.5})))
      self._eval_config = tfma.EvalConfig(
          model_specs=[tfma.ModelSpec(label_key="tips")],
          metrics_specs=metric_specs)
      self._eval_shared_models = {
          "":
              tfma.default_eval_shared_model(
                  self._dataset.trained_saved_model_path(),
                  eval_config=self._eval_config)
      }

  def _max_num_examples(self):
    # TFMA is slower than TFT, so use a smaller number of examples from the
    # dataset.
    limit = 100000
    parent_max = super()._max_num_examples()
    if parent_max is None:
      return limit
    return min(parent_max, limit)

  def report_benchmark(self, **kwargs):
    if "extras" not in kwargs:
      kwargs["extras"] = {}
    # Note that the GIT_COMMIT_ID is not included in the packages themselves:
    # it must be injected by an external script.
    kwargs["extras"]["commit_tfx"] = (
        getattr(tfx, "GIT_COMMIT_ID", None) or
        getattr(tfx, "__version__", None))
    kwargs["extras"]["commit_tfma"] = (
        getattr(tfma, "GIT_COMMIT_ID", None) or
        getattr(tfma, "__version__", None))
    # Stdout for use in tools which read the benchmark results from stdout.
    print(self._get_name(), kwargs["wall_time"],
          "({}x)".format(kwargs["iters"]))
    super().report_benchmark(**kwargs)

  def _runMiniPipeline(self, multi_model):
    """Benchmark a "mini" TFMA - predict, slice and compute metrics.

    Runs a "mini" version of TFMA in a Beam pipeline. Records the wall time
    taken for the whole pipeline.

    Args:
      multi_model: True if multiple models should be used in the benchmark.
    """
    self._init_model(multi_model, validation=False)
    pipeline = self._create_beam_pipeline()
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=benchmark_utils.read_schema(
            self._dataset.tf_metadata_schema_path()),
        raw_record_column_name=constants.ARROW_INPUT_COLUMN)
    raw_data = (
        pipeline
        | "Examples" >> beam.Create(
            self._dataset.read_raw_dataset(
                deserialize=False, limit=self._max_num_examples()))
        | "BatchExamples" >> tfx_io.BeamSource()
        | "InputsToExtracts" >> tfma.BatchedInputsToExtracts())

    def rescale_labels(extracts):
      # Transform labels to [0, 1] so we can test metrics that require labels in
      # that range.
      result = copy.copy(extracts)
      result[constants.LABELS_KEY] = self._transform_labels(
          extracts[constants.LABELS_KEY])
      return result

    _ = (
        raw_data
        | "FeaturesExtractor" >> features_extractor.FeaturesExtractor(
            eval_config=self._eval_config).ptransform
        | "LabelsExtractor" >> labels_extractor.LabelsExtractor(
            eval_config=self._eval_config).ptransform
        | "RescaleLabels" >> beam.Map(rescale_labels)
        | "ExampleWeightsExtractor" >> example_weights_extractor
        .ExampleWeightsExtractor(eval_config=self._eval_config).ptransform
        | "PredictionsExtractor" >> predictions_extractor.PredictionsExtractor(
            eval_config=self._eval_config,
            eval_shared_model=self._eval_shared_models).ptransform
        | "UnbatchExtractor" >> unbatch_extractor.UnbatchExtractor().ptransform
        | "SliceKeyExtractor" >> tfma.extractors.SliceKeyExtractor().ptransform
        | "ComputeMetricsPlotsAndValidations" >>
        metrics_plots_and_validations_evaluator
        .MetricsPlotsAndValidationsEvaluator(
            eval_config=self._eval_config,
            eval_shared_model=self._eval_shared_models).ptransform)

    start = time.time()
    for _ in range(_ITERS):
      result = pipeline.run()
      result.wait_until_finish()
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=_ITERS,
        wall_time=delta,
        extras={
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkMiniPipeline(self):
    self._runMiniPipeline(False)

  def benchmarkMiniPipelineMultiModel(self):
    self._runMiniPipeline(True)

  def _readDatasetIntoExtracts(self):
    """Read the raw dataset and massage examples into Extracts."""
    records = []
    for x in self._dataset.read_raw_dataset(
        deserialize=False, limit=self._max_num_examples()):
      records.append({tfma.INPUT_KEY: x, tfma.SLICE_KEY_TYPES_KEY: ()})
    return records

  def _readDatasetIntoBatchedExtracts(self):
    """Read the raw dataset and massage examples into batched Extracts."""
    serialized_examples = list(
        self._dataset.read_raw_dataset(
            deserialize=False, limit=self._max_num_examples()))

    # TODO(b/153996019): Once the TFXIO interface that returns an iterator of
    # RecordBatch is available, clean this up.
    coder = example_coder.ExamplesToRecordBatchDecoder(
        serialized_schema=benchmark_utils.read_schema(
            self._dataset.tf_metadata_schema_path()).SerializeToString())
    batches = []
    for i in range(0, len(serialized_examples), _BATCH_SIZE):
      example_batch = serialized_examples[i:i + _BATCH_SIZE]
      record_batch = record_based_tfxio.AppendRawRecordColumn(
          coder.DecodeBatch(example_batch), constants.ARROW_INPUT_COLUMN,
          example_batch)
      batches.append({constants.ARROW_RECORD_BATCH_KEY: record_batch})
    return batches

  def _transform_labels(self, labels):
    # Transform labels to [0, 1] so we can test metrics that require labels in
    # that range.
    if len(self._eval_config.model_specs) > 1:
      updated_labels = {}
      for s in self._eval_config.model_specs:
        updated_labels[s.name] = np.array(
            [1.0 / (1.0 + x) for x in labels[s.name]])
      return updated_labels
    else:
      return np.array([1.0 / (1.0 + x) for x in labels])

  def _extract_features_and_labels(self, batched_extract):
    """Extract features from extracts containing arrow table."""
    # This function is a combination of
    # _ExtractFeatures.extract_features in extractors/features_extractor.py
    # and _ExtractLabels.extract_labels in extractors/labels_extractor.py
    result = copy.copy(batched_extract)
    (record_batch, serialized_examples) = (
        features_extractor._drop_unsupported_columns_and_fetch_raw_data_column(  # pylint: disable=protected-access
            batched_extract[constants.ARROW_RECORD_BATCH_KEY]))
    features = result[
        constants.FEATURES_KEY] if constants.FEATURES_KEY in result else {}
    features.update(util.record_batch_to_tensor_values(record_batch))
    result[constants.FEATURES_KEY] = features
    result[constants.INPUT_KEY] = serialized_examples
    labels = (
        model_util.get_feature_values_for_model_spec_field(
            list(self._eval_config.model_specs), "label_key", "label_keys",
            result, True))
    result[constants.LABELS_KEY] = self._transform_labels(labels)
    return result

  def _runInputExtractorManualActuation(self, multi_model):
    """Benchmark InputExtractor "manually"."""
    self._init_model(multi_model, validation=False)
    records = self._readDatasetIntoExtracts()
    extracts = []

    start = time.time()
    for _ in range(_ITERS):
      for elem in records:
        extracts.append(
            legacy_input_extractor._ParseExample(elem, self._eval_config))  # pylint: disable=protected-access
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=_ITERS, wall_time=delta, extras={"num_examples": len(records)})

  # "Manual" micro-benchmarks
  def benchmarkInputExtractorManualActuation(self):
    self._runInputExtractorManualActuation(False)

  # "Manual" micro-benchmarks
  def benchmarkInputExtractorManualActuationMultiModel(self):
    self._runInputExtractorManualActuation(True)

  def _runFeaturesExtractorManualActuation(self, multi_model):
    """Benchmark FeaturesExtractor "manually"."""
    self._init_model(multi_model, validation=False)
    extracts = self._readDatasetIntoBatchedExtracts()
    num_examples = sum(
        [e[constants.ARROW_RECORD_BATCH_KEY].num_rows for e in extracts])
    result = []
    start = time.time()
    for _ in range(_ITERS):
      for e in extracts:
        result.append(self._extract_features_and_labels(e))
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=_ITERS, wall_time=delta, extras={"num_examples": num_examples})

  # "Manual" micro-benchmarks
  def benchmarkFeaturesExtractorManualActuation(self):
    self._runFeaturesExtractorManualActuation(False)

  # "Manual" micro-benchmarks
  def benchmarkFeaturesExtractorManualActuationMultiModel(self):
    self._runFeaturesExtractorManualActuation(True)

  def _runPredictionsExtractorManualActuation(self, multi_model):
    """Benchmark PredictionsExtractor "manually"."""
    self._init_model(multi_model, validation=False)
    extracts = self._readDatasetIntoBatchedExtracts()
    num_examples = sum(
        [e[constants.ARROW_RECORD_BATCH_KEY].num_rows for e in extracts])
    extracts = [self._extract_features_and_labels(e) for e in extracts]

    prediction_do_fn = model_util.ModelSignaturesDoFn(
        eval_config=self._eval_config,
        eval_shared_models=self._eval_shared_models,
        signature_names={
            constants.PREDICTIONS_KEY: {
                name: [None] for name in self._eval_shared_models
            }
        },
        prefer_dict_outputs=False)
    prediction_do_fn.setup()

    start = time.time()
    for _ in range(_ITERS):
      predict_result = []
      for e in extracts:
        predict_result.extend(prediction_do_fn.process(e))
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=_ITERS, wall_time=delta, extras={"num_examples": num_examples})

  # "Manual" micro-benchmarks
  def benchmarkPredictionsExtractorManualActuation(self):
    self._runPredictionsExtractorManualActuation(False)

  # "Manual" micro-benchmarks
  def benchmarkPredictionsExtractorManualActuationMultiModel(self):
    self._runPredictionsExtractorManualActuation(True)

  def _runMetricsPlotsAndValidationsEvaluatorManualActuation(
      self,
      with_confidence_intervals,
      multi_model,
      metrics_specs=None,
      validation=False):
    """Benchmark MetricsPlotsAndValidationsEvaluator "manually"."""
    self._init_model(multi_model, validation)
    if not metrics_specs:
      metrics_specs = self._eval_config.metrics_specs

    extracts = self._readDatasetIntoBatchedExtracts()
    num_examples = sum(
        [e[constants.ARROW_RECORD_BATCH_KEY].num_rows for e in extracts])
    extracts = [self._extract_features_and_labels(e) for e in extracts]

    prediction_do_fn = model_util.ModelSignaturesDoFn(
        eval_config=self._eval_config,
        eval_shared_models=self._eval_shared_models,
        signature_names={
            constants.PREDICTIONS_KEY: {
                name: [None] for name in self._eval_shared_models
            }
        },
        prefer_dict_outputs=False)
    prediction_do_fn.setup()

    # Have to predict first
    predict_result = []
    for e in extracts:
      predict_result.extend(prediction_do_fn.process(e))

    # Unbatch extracts
    unbatched_extracts = []
    for e in predict_result:
      unbatched_extracts.extend(unbatch_extractor._extract_unbatched_inputs(e))  # pylint: disable=protected-access

    # Add global slice key.
    for e in unbatched_extracts:
      e[tfma.SLICE_KEY_TYPES_KEY] = ()

    # Now Evaluate
    inputs_per_accumulator = 1000
    start = time.time()
    for _ in range(_ITERS):
      computations, _, _, _ = (
          # pylint: disable=protected-access
          metrics_plots_and_validations_evaluator
          ._filter_and_separate_computations(
              metric_specs_util.to_computations(
                  metrics_specs, eval_config=self._eval_config)))
      # pylint: enable=protected-access

      processed = []
      for elem in unbatched_extracts:
        processed.append(
            next(
                metrics_plots_and_validations_evaluator._PreprocessorDoFn(  # pylint: disable=protected-access
                    computations).process(elem)))

      combiner = metrics_plots_and_validations_evaluator._ComputationsCombineFn(  # pylint: disable=protected-access
          computations=computations)
      if with_confidence_intervals:
        combiner = poisson_bootstrap._BootstrapCombineFn(combiner)  # pylint: disable=protected-access
      combiner.setup()

      accumulators = []
      for batch in benchmark_utils.batched_iterator(processed,
                                                    inputs_per_accumulator):
        accumulator = combiner.create_accumulator()
        for elem in batch:
          accumulator = combiner.add_input(accumulator, elem)
        accumulators.append(accumulator)

      final_accumulator = combiner.merge_accumulators(accumulators)
      final_output = combiner.extract_output(final_accumulator)
    end = time.time()
    delta = end - start

    # Sanity check the example count. This is not timed.
    example_count_key = metric_types.MetricKey(
        name="example_count", model_name="candidate" if multi_model else "")
    if example_count_key in final_output:
      example_count = final_output[example_count_key]
    else:
      raise ValueError("example_count_key ({}) was not in the final list of "
                       "metrics. metrics were: {}".format(
                           example_count_key, final_output))

    if with_confidence_intervals:
      # If we're computing using confidence intervals, the example count will
      # not be exact.
      lower_bound = int(0.9 * num_examples)
      upper_bound = int(1.1 * num_examples)
      if example_count < lower_bound or example_count > upper_bound:
        raise ValueError("example count out of bounds: expecting "
                         "%d < example_count < %d, but got %d" %
                         (lower_bound, upper_bound, example_count))
    else:
      # If we're not using confidence intervals, we expect the example count to
      # be exact.
      if example_count != num_examples:
        raise ValueError("example count mismatch: expecting %d got %d" %
                         (num_examples, example_count))

    self.report_benchmark(
        iters=_ITERS,
        wall_time=delta,
        extras={
            "inputs_per_accumulator": inputs_per_accumulator,
            "num_examples": num_examples
        })

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorManualActuationNoConfidenceIntervals(
      self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False, multi_model=False, validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorManualActuationNoConfidenceIntervalsMultiModel(
      self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False, multi_model=True, validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorManualActuationWithConfidenceIntervals(
      self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=True, multi_model=False, validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorManualActuationWithConfidenceIntervalsMultiModel(
      self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=True, multi_model=True, validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorAUC10k(self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False,
        multi_model=False,
        metrics_specs=metric_specs_util.specs_from_metrics([
            tf.keras.metrics.AUC(name="auc", num_thresholds=10000),
        ]),
        validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorAUC10kMultiModel(self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False,
        multi_model=True,
        metrics_specs=metric_specs_util.specs_from_metrics(
            [
                tf.keras.metrics.AUC(name="auc", num_thresholds=10000),
            ],
            model_names=["candidate", "baseline"]),
        validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorBinaryClassification(self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False,
        multi_model=False,
        metrics_specs=metric_specs_util.specs_from_metrics([
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", num_thresholds=10000),
            tf.keras.metrics.AUC(
                name="auc_precison_recall", curve="PR", num_thresholds=10000),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tfma.metrics.MeanLabel(name="mean_label"),
            tfma.metrics.MeanPrediction(name="mean_prediction"),
            tfma.metrics.Calibration(name="calibration"),
            tfma.metrics.ConfusionMatrixPlot(name="confusion_matrix_plot"),
            tfma.metrics.CalibrationPlot(name="calibration_plot"),
        ]),
        validation=True)

  # "Manual" micro-benchmarks
  def benchmarkMetricsPlotsAndValidationsEvaluatorBinaryClassificationMultiModel(
      self):
    self._runMetricsPlotsAndValidationsEvaluatorManualActuation(
        with_confidence_intervals=False,
        multi_model=True,
        metrics_specs=metric_specs_util.specs_from_metrics([
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", num_thresholds=10000),
            tf.keras.metrics.AUC(
                name="auc_precison_recall", curve="PR", num_thresholds=10000),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tfma.metrics.MeanLabel(name="mean_label"),
            tfma.metrics.MeanPrediction(name="mean_prediction"),
            tfma.metrics.Calibration(name="calibration"),
            tfma.metrics.ConfusionMatrixPlot(name="confusion_matrix_plot"),
            tfma.metrics.CalibrationPlot(name="calibration_plot"),
        ],
                                                           model_names=[
                                                               "candidate",
                                                               "baseline"
                                                           ]),
        validation=True)
