# Copyright 2019 Google LLC. All Rights Reserved.
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
"""TFMA benchmark."""

import time


import apache_beam as beam
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.eval_saved_model import load

import tfx
from tfx.benchmarks import benchmark_utils
from tfx.benchmarks import benchmark_base


class TFMABenchmarkBase(benchmark_base.BenchmarkBase):
  """TFMA benchmark base class."""

  def __init__(self, dataset, **kwargs):
    # Benchmark runners may pass extraneous arguments we don't care about.
    del kwargs
    super().__init__()
    self._dataset = dataset

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
    kwargs["extras"]["commit_tfx"] = (getattr(tfx, "GIT_COMMIT_ID", None) or
                                      getattr(tfx, "__version__", None))
    kwargs["extras"]["commit_tfma"] = (getattr(tfma, "GIT_COMMIT_ID", None) or
                                       getattr(tfma, "__version__", None))
    super().report_benchmark(**kwargs)

  def benchmarkMiniPipeline(self):
    """Benchmark a "mini" version of TFMA - predict, slice and compute metrics.

    Runs a "mini" version of TFMA in a Beam pipeline. Records the wall time
    taken for the whole pipeline.
    """
    pipeline = self._create_beam_pipeline()
    raw_data = (
        pipeline
        | "Examples" >> beam.Create(
            self._dataset.read_raw_dataset(
                deserialize=False, limit=self._max_num_examples()))
        | "InputsToExtracts" >> tfma.InputsToExtracts())

    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=self._dataset.tfma_saved_model_path())

    _ = (
        raw_data
        | "PredictExtractor" >> tfma.extractors.PredictExtractor(
            eval_shared_model=eval_shared_model).ptransform
        | "SliceKeyExtractor" >> tfma.extractors.SliceKeyExtractor().ptransform
        | "ComputeMetricsAndPlots" >> tfma.evaluators.MetricsAndPlotsEvaluator(
            eval_shared_model=eval_shared_model).ptransform)

    start = time.time()
    result = pipeline.run()
    result.wait_until_finish()
    end = time.time()
    delta = end - start

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkPredict(self):
    """Benchmark the predict and aggregate combine stages "manually".

    Runs _TFMAPredictionDoFn "manually" outside a Beam pipeline. Records the
    wall time taken.
    """
    # Run InputsToExtracts manually.
    records = []
    for x in self._dataset.read_raw_dataset(
        deserialize=False, limit=self._max_num_examples()):
      records.append({tfma.constants.INPUT_KEY: x})

    fn = tfma.extractors.legacy_predict_extractor._TFMAPredictionDoFn(  # pylint: disable=protected-access
        eval_shared_models={"": tfma.default_eval_shared_model(
            eval_saved_model_path=self._dataset.tfma_saved_model_path())},
        eval_config=None)
    fn.setup()

    # Predict
    predict_batch_size = 1000
    predict_result = []
    start = time.time()
    for batch in benchmark_utils.batched_iterator(records, predict_batch_size):
      predict_result.extend(fn.process(batch))
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                predict_batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkAggregateCombineManualActuation(self):
    """Benchmark the aggregate combine stage "manually".

    Runs _AggregateCombineFn "manually" outside a Beam pipeline. Records the
    wall time taken.
    """

    # Run InputsToExtracts manually.
    records = []
    for x in self._dataset.read_raw_dataset(
        deserialize=False, limit=self._max_num_examples()):
      records.append({tfma.constants.INPUT_KEY: x})

    fn = tfma.extractors.legacy_predict_extractor._TFMAPredictionDoFn(  # pylint: disable=protected-access
        eval_shared_models={"": tfma.default_eval_shared_model(
            eval_saved_model_path=self._dataset.tfma_saved_model_path())},
        eval_config=None)
    fn.setup()

    # Predict
    predict_batch_size = 1000
    predict_result = []
    for batch in benchmark_utils.batched_iterator(records, predict_batch_size):
      predict_result.extend(fn.process(batch))

    # AggregateCombineFn
    #
    # We simulate accumulating records into multiple different accumulators,
    # each with inputs_per_accumulator records, and then merging the resulting
    # accumulators together at one go.

    # Number of elements to feed into a single accumulator.
    # (This means we will have len(records) / inputs_per_accumulator
    # accumulators to merge).
    inputs_per_accumulator = 1000

    combiner = tfma.evaluators.legacy_aggregate._AggregateCombineFn(  # pylint: disable=protected-access
        eval_shared_model=tfma.default_eval_shared_model(
            eval_saved_model_path=self._dataset.tfma_saved_model_path()))
    combiner.setup()
    accumulators = []

    start = time.time()
    for batch in benchmark_utils.batched_iterator(predict_result,
                                                  inputs_per_accumulator):
      accumulator = combiner.create_accumulator()
      for elem in batch:
        combiner.add_input(accumulator, elem)
      accumulators.append(accumulator)
    final_accumulator = combiner.merge_accumulators(accumulators)
    final_output = combiner.extract_output(final_accumulator)
    end = time.time()
    delta = end - start

    # Extract output to sanity check example count. This is not timed.
    extract_fn = tfma.evaluators.legacy_aggregate._ExtractOutputDoFn(  # pylint: disable=protected-access
        eval_shared_model=tfma.default_eval_shared_model(
            eval_saved_model_path=self._dataset.tfma_saved_model_path()))
    extract_fn.setup()
    interpreted_output = list(extract_fn.process(((), final_output)))
    if len(interpreted_output) != 1:
      raise ValueError("expecting exactly 1 interpreted output, got %d" %
                       (len(interpreted_output)))
    got_example_count = interpreted_output[0][1].get(
        "post_export_metrics/example_count")
    if got_example_count != self._dataset.num_examples(
        limit=self._max_num_examples()):
      raise ValueError(
          "example count mismatch: expecting %d got %d" %
          (self._dataset.num_examples(limit=self._max_num_examples()),
           got_example_count))

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "inputs_per_accumulator":
                inputs_per_accumulator,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkEvalSavedModelPredict(self):
    """Benchmark using the EvalSavedModel to make predictions.

    Runs EvalSavedModel.predict_list and records the wall time taken.
    """
    batch_size = 1000

    eval_saved_model = load.EvalSavedModel(
        path=self._dataset.tfma_saved_model_path(),
        include_default_metrics=True)

    records = self._dataset.read_raw_dataset(
        deserialize=False, limit=self._max_num_examples())

    start = time.time()
    for batch in benchmark_utils.batched_iterator(records, batch_size):
      eval_saved_model.predict_list(batch)
    end = time.time()
    delta = end - start
    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })

  def benchmarkEvalSavedModelMetricsResetUpdateGetList(self):
    """Benchmark using the EvalSavedModel to compute metrics.

    Runs EvalSavedModel.metrics_reset_update_get_list and records the wall time
    taken.
    """
    batch_size = 1000

    eval_saved_model = load.EvalSavedModel(
        path=self._dataset.tfma_saved_model_path(),
        include_default_metrics=True)

    records = self._dataset.read_raw_dataset(
        deserialize=False, limit=self._max_num_examples())

    start = time.time()
    accumulators = []
    for batch in benchmark_utils.batched_iterator(records, batch_size):
      accumulators.append(eval_saved_model.metrics_reset_update_get_list(batch))
    end = time.time()
    delta = end - start

    # Sanity check
    metric_variables_sum = accumulators[0]
    for acc in accumulators[1:]:
      if len(metric_variables_sum) != len(acc):
        raise ValueError(
            "all metric variable value lists should have the same length, but "
            "got lists with different lengths: %d and %d" %
            (len(metric_variables_sum), len(acc)))
      metric_variables_sum = [a + b for a, b in zip(metric_variables_sum, acc)]

    metrics = eval_saved_model.metrics_set_variables_and_get_values(
        metric_variables_sum)
    if "average_loss" not in metrics:
      raise ValueError(
          "metrics should contain average_loss metric, but it did not. "
          "metrics were: %s" % metrics)

    self.report_benchmark(
        iters=1,
        wall_time=delta,
        extras={
            "batch_size":
                batch_size,
            "num_examples":
                self._dataset.num_examples(limit=self._max_num_examples())
        })
