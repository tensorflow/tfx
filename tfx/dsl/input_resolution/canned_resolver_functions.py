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
"""Module for public facing, canned resolver functions."""

from typing import Sequence, Optional

from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.types import channel as channel_type
from tfx.types import standard_artifacts


@resolver_function.resolver_function
def latest_created(artifacts, n: int = 1):
  """Returns the n latest createst artifacts, ties broken by artifact id.

  Args:
    artifacts: The artifacts to filter.
    n: The number of latest artifacts to return, must be > 0.

  Returns:
    The n latest artifacts.
  """
  return ops.LatestCreateTime(artifacts, n=n)


@resolver_function.resolver_function
def latest_version(artifacts, n: int = 1):
  """Returns the n latest version artifacts, ties broken by artifact id.

  Args:
    artifacts: The artifacts to filter.
    n: The number of latest artifacts to return, must be > 0.

  Returns:
    The n latest artifacts.
  """
  return ops.LatestVersion(artifacts, n=n)


@resolver_function.resolver_function
def static_range(artifacts,
                 *,
                 start_span_number: int = -1,
                 end_span_number: int = -1,
                 keep_all_versions: bool = False,
                 exclude_span_numbers: Sequence[int] = (),
                 min_spans: Optional[int] = None):
  """Returns artifacts with spans in [start_span, end_span] inclusive.

  This resolver function is based on the span-version semantics, which only
  considers the latest version of each span. If you want to keep all versions,
  then set keep_all_versions=True. Input artifacts must have both "span" int
  property and "version" int property.

  Please note that the spans in exclude_span_numbers are excluded AFTER getting
  the artifacts with spans in the range.

  If there are less than min_spans unique spans present in the resolved
  artifacts, then the component execution will be skipped.

  Corresponds to StaticRange in TFX.

  Example usage:

    Consider 8 artifacts with:
      spans    = [0, 1, 2, 3, 3, 5, 7, 10]
      versions = [0, 0, 0, 0, 3, 0, 0, 0]

    static_range(
      end_span_number=5,
      keep_all_versions=False,
      exclude_span_numbers=[2])

    Because start_span_number = -1, it is set to the smallest span, 0.

    Spans in the range [0, 5] will be considered.

    Because keep_all_versions=False, only the artifact with span=3 and version=3
    will be considered, even though there are two artifacts with span=3.

    Because exclude_span_numbers=[2], the artifacts with span=2 will not be
    kept, even though it is in the range.

    min_spans is None but end_span_number < 0, so min_spans is not automatically
    set.

    The artifacts that will be returned are:
      spans    = [0, 1, 3, 5]
      versions = [0, 0, 3, 0]

  Args:
    artifacts: The artifacts to filter.
    start_span_number: The smallest span number to keep, inclusive. If < 0, set
      to the smallest span in the artifacts.
    end_span_number: The largest span number to keep, inclusive. If < 0, set to
      the largest span in the artifacts.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.
    exclude_span_numbers: The span numbers to exclude.
    min_spans: Minimum number of desired example spans in the range. If
      min_spans is None, and if both end_span_number and start_span_number are
      positive, it is set to end_span_number - start_span_number + 1. Else if
      min_spans is None, it is set to -1.

  Returns:
    Artifacts with spans in [start_span, end_span] inclusive.
  """
  resolved_artifacts = ops.StaticSpanRange(
      artifacts,
      start_span=start_span_number,
      end_span=end_span_number,
      keep_all_versions=keep_all_versions)
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers)

  if min_spans is None:
    # We check that start_span_number and end_span_number are positive to ensure
    # min_spans is well defined. Else, it is set to -1, meaning all the unique
    # spans will be considered.
    if start_span_number >= 0 and end_span_number >= 0:
      min_spans = end_span_number - start_span_number + 1
    else:
      min_spans = -1

  return ops.SkipIfLessThanNSpans(resolved_artifacts, n=min_spans)


@resolver_function.resolver_function
def rolling_range(artifacts,
                  *,
                  start_span_number: int = 0,
                  num_spans: int = 1,
                  skip_num_recent_spans: int = 0,
                  keep_all_versions: bool = False,
                  exclude_span_numbers: Sequence[int] = (),
                  min_spans: Optional[int] = None):
  """Returns artifacts with spans in a rolling range.

  A rolling range covers the latest (largest) spans. It's calculated in the
  following order:

  1. Sort the artifacts by span in ascending order.
  2. Remove the last skip_num_recent_spans number of spans (removing the largest
     spans).
  3. Select the last num_spans number of spans (the remaining largest spans).
  4. Exclude the spans of exclude_span_numbers. Note that this exclusion
     happens last for backward compatibility. This can result in having less
     than num_spans spans, meaning the consumer component would be skipped due
     to lack of inputs. To avoid this, you would have to increase min_spans.

  Pythonically, this range is equivalent to:
  sorted_spans[:-skip_num_recent_spans][-num_spans:]

  This resolver function is based on the span-version semantics, which only
  considers the latest version of each span. If you want to keep all versions,
  then set keep_all_versions=True. Input artifacts must have both "span" int
  property and "version" int property.

  Please note that the spans in exclude_span_numbers are excluded AFTER getting
  the latest spans.

  If there are less than min_spans unique spans present in the resolved
  artifacts, then the component execution will be skipped.

  Corresponds to RollingRange in TFX.

  Example usage:

    Consider 6 artifacts with:
      spans    = [1, 2, 3, 3, 7, 8]
      versions = [0, 0, 1, 0, 1, 2]

    rolling_range(
        start_span_number=3,
        num_spans=5,
        skip_num_recent_spans=1,
        keep_all_versions=True,
        exclude_span_numbers=[7],
        min_spans=1)

    spans 1 and 2 are removed because they are < start_span_number=3. The
    sorted unique spans are [3, 7, 8].

    span 8 is removed because skip_num_recent_spans=1, leaving spans [3, 7].

    Although num_spans=5, only two unique span numbers are available, 3 and 7,
    so both spans [3, 7] are kept.

    Because keep_all_versions=True, both artifacts with span=3 are kept.

    Because exclude_span_numbers=[7], the artifact with span=7 will not be
    kept, even though it is in the range.

    The artifacts that will be returned are:
      spans = [3, 3]
      versions = [1, 0]

    Note min_spans=1, so a SkipSignal will not be present in the compiled IR.

  Args:
    artifacts: The artifacts to filter.
    start_span_number: The smallest span number to keep, inclusive. Defaults to
      0.
    num_spans: The length of the range. If num_spans <= 0, then num_spans is set
      to the total number of unique spans.
    skip_num_recent_spans: Number of most recently available (largest) spans to
      skip. Defaults to 0.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.
    exclude_span_numbers: The span numbers to exclude.
    min_spans: Minimum number of desired example spans in the range. If
      min_spans is None, it is set to num_spans.

  Returns:
    Artifacts with spans in the rolling range.
  """
  resolved_artifacts = ops.LatestSpan(
      artifacts,
      min_span=start_span_number,
      n=num_spans,
      skip_last_n=skip_num_recent_spans,
      keep_all_versions=keep_all_versions)
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers)

  if min_spans is None:
    min_spans = num_spans

  return ops.SkipIfLessThanNSpans(resolved_artifacts, n=min_spans)


@resolver_function.resolver_function
def all_spans(artifacts):
  """Returns the sorted artifacts with unique spans.

  By default, all artifacts with unique spans (ties broken by latest version)
  are returned.

  Example usage:

    Consider 6 artifacts with:
      spans    = [1, 3, 3, 2, 8, 7]
      versions = [0, 0, 1, 0, 1, 2]

    all_spans()

    will return artifacts:
      spans    = [1, 2, 3, 7, 8]
      versions = [0, 0, 1, 2, 1]

    Note that there are 2 artifacts with span 3, but only the one with the
    latest version is returned. Spans are sorted in ascending order.

  Args:
    artifacts: The artifacts to filter.

  Returns:
    Sorted Artifacts with unique spans.
  """
  return ops.AllSpans(artifacts)


@resolver_function.resolver_function
def shuffle(artifacts):
  """Shuffles the artifacts (in a uniform random way) by span.

  Example usage:

  Consider 4 artifacts with:
    spans    = [1, 2, 3, 4]

  shuffle()

  will return artifacts randomly shuffled, e.g.:
    spans = [3, 4, 2, 1]

  Args:
    artifacts: The artifacts to filter.

  Returns:
    The randomly shuffled artifacts.
  """
  return ops.Shuffle(artifacts)


@resolver_function.resolver_function
def latest_pipeline_run_outputs(pipeline, output_keys: Sequence[str] = ()):
  """Returns the artifacts in the latest COMPLETE pipeline run.

  Example usage:

    producer_pipeline = Pipeline(outputs={
        'examples': example_gen.outputs['examples'],
        'schema': schema_gen.outputs['schema']
    })

    consumer_pipeline_inputs = PipelineInputs(
        latest_pipeline_run_outputs(producer_pipeline),
        output_keys=['examples', 'schema'])
    trainer = TFTrainer(
        examples=pipeline_inputs.inputs['examples'],
        schema=pipeline_inputs.inputs['schema'])
    consumer_pipeline = Pipeline(
        inputs=consumer_pipeline_inputs,
        components=[trainer],
    )

  Args:
    pipeline: The pipeline producing the artifacts
    output_keys: (Optional) A list of output keys. If provided, only the
      artifacts of the key in this list will return by this function, otherwise,
      all available output keys of the producer pipeline will be used.

  Returns:
    The artifacts in the latest COMPLETE pipeline run.
  """
  for output_key in output_keys:
    if output_key not in pipeline.outputs:
      raise ValueError(
          f'Output key {output_key} does not exist in pipeline {pipeline.id}. '
          f'Available: {list(pipeline.outputs)}')
  return ops.LatestPipelineRunOutputs(
      pipeline_name=pipeline.pipeline_name, output_keys=output_keys)


@latest_pipeline_run_outputs.output_type_inferrer
def _infer_latest_pipeline_run_type(pipeline, output_keys: Sequence[str] = ()):
  """Output type inferrer of resolver function latest_pipeline_run_outputs.

  Args:
    pipeline: The pipeline producing the artifacts.
    output_keys: (Optional) A list of output keys. If provided, only the
      artifacts of the key in this list will return by this function, otherwise,
      all available output keys of the producer pipeline will be used.

  Returns:
    A Dict: key is output key, value is output type.
  """
  if not output_keys:
    output_keys = list(pipeline.outputs)
  return {
      output_key: channel.type
      for output_key, channel in pipeline.outputs.items()
      if output_key in output_keys
  }


@resolver_function.resolver_function(unwrap_dict_key='window')
def sequential_rolling_range(artifacts,
                             *,
                             start_span_number: Optional[int] = None,
                             num_spans: int = 1,
                             skip_num_recent_spans: int = 0,
                             keep_all_versions: bool = False,
                             exclude_span_numbers: Sequence[int] = ()):
  """Returns artifacts with spans in a sequential rolling range.

  Sequential rolling range is a sliding window on the oldest consecutive spans.

  The consecutive spans must be in the range:
  [start_span_number, max_span - skip_num_recent_spans], where max_span is the
  maximum span present in the artifacts. This range is modified to account for
  exclude_span_numbers, for details see the ConsecutiveSpans ResolverOp
  implementation.

  The window size is num_spans and has a stride of 1. If the spans are not
  consecutive, then the sequential rolling range waits for the missing span to
  arrive.

  This resolver function is based on the span-version semantics, which only
  considers the latest version of each span. If you want to keep all versions,
  then set keep_all_versions=True. Input artifacts must have both "span" int
  property and "version" int property.

  Corresponds to SequentialRollingRange in TFX.

  Example usage:

    Consider 5 artifacts [A, B, C, D, E] with spans = [1, 2, 3, 4, 7].

    sequential_rolling_range(
        start_span_number=1,
        num_spans=3,
        skip_num_recent_spans=1,
        keep_all_versions=False,
        exclude_span_numbers=[])

    The consecutive spans to consider are [1, 2, 3, 4]

    The artifacts will be returned with a sliding window of size num_spans=3 and
    stride 1 applied:

    [[A, B, C], [B, C, D]]

    However, if nums_spans=5, there are only 4 consecutive spans to consider, so
    [], no artifacts, will be returned.

  Since sequential_rolling_range returns multiple windows, it must be used
  together with ForEach. For example:

    with ForEach(sequential_rolling_range(
        all_examples, num_spans=10)) as examples_window:
      trainer = Trainer(examples=shuffle(examples_window))

  Args:
    artifacts: The artifacts to filter.
    start_span_number: The smallest span number to keep, inclusive. Optional, if
      not set then defaults to the minimum span. If the start_span_number is
      configured wrong (so that it is smaller than the first span number), we
      will wait indefinitely until the missing spans between start_span_number
      and the first span number to be appeared.
    num_spans: The length of the range. If num_spans <= 0, then num_spans is set
      to the total number of artifacts with consecutive spans in the range. Note
      that this is also the size of the sliding window of the sequential rolling
      range.
    skip_num_recent_spans: Number of most recently available (largest) spans to
      skip. Defaults to 0.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.
    exclude_span_numbers: The list of missing/bad span numbers to exclude.

  Returns:
    Artifacts with spans in the sequential rolling range.
  """
  resolved_artifacts = ops.ConsecutiveSpans(
      artifacts,
      first_span=start_span_number if start_span_number is not None else -1,
      skip_last_n=skip_num_recent_spans,
      keep_all_versions=keep_all_versions,
      denylist=exclude_span_numbers)

  return ops.SlidingWindow(resolved_artifacts, window_size=num_spans)


@sequential_rolling_range.output_type_inferrer
def _infer_seqential_rolling_range_type(channel, **kwargs):  # pylint: disable=unused-argument
  return {'window': channel.type}


@resolver_function.resolver_function(output_type=standard_artifacts.Examples)
def training_range(model):
  """Returns the Examples artifacts that a Model was trained on.

  Let there be [Examples 1, Examples 2, Examples 3, Examples 4] present in MLMD,
  as well as [Model 1, Model 2, Model 3].

  Also let Model 1 be trained on [Examples 2, Examples 3].

  training_range() called on Model 1 will return [Examples 2, Examples 3].

  Example Usage:

    latest_trained_model = latest_trained(trainer.outputs["model"])
    with ForEach(training_range(latest_trained_model)) as each_examples:
      bulk_inferrer = BulkInferrer(
          examples=each_examples.no_trigger(),
          model=latest_trained_model,
          ...
      )

    Note that BulkInferrer can only process a single Example at a time, hence
    the ForEach.

  Args:
    model: The Model artifact to find training Examples of.

  Returns:
    The Examples artifacts used to train the Model.
  """
  return ops.TrainingRange(model)


@resolver_function.resolver_function
def span_driven_evaluator_inputs(
    examples: channel_type.BaseChannel,
    models: channel_type.BaseChannel,
    wait_spans_before_eval: int = 0,
    evaluation_training_offset: int = 0,
    additional_spans_per_eval: int = 0,
    start_span_number: int = 0,
):
  """Returns a dictionary with a Model and the Examples to evaluate it with.

  ******************************************************************************
  INTENDED FOR TFX -> TFLEX MIGRATION ONLY, CONSULT TFX TEAM BEFORE USE.
  ******************************************************************************
  See ContinuousSpanDrivenSpec in TFX for details.

  Let max_span be the largest span present in the passed in examples.

  The Examples with spans > start_span_number and in the following range
  (inclusive) will be used to evaluate the Model:

  [
    max_span - wait_spans_before_eval - additional_spans_per_eval
    max_span - wait_spans_before_eval
  ]

  Let start_span = max_span - wait_spans_before_eval - additional_spans_per_eval

  Then, the latest created Model trained on a span smaller than (exclusive)
  start_span - evaluation_training_offset will be selected.

  Consider Examples with spans [1, 2, 3, ..., 10]. And let there be 7 Models,
  where Model N was trained on spans [N, N+1, N+2]. So Model 4 was trained on
  spans [3, 4, 5].

  span_driven_evaluator_inputs(
    examples,
    models,
    wait_spans_before_eval=1,
    evaluation_training_offset=2,
    additional_spans_per_eval=3
    start_span_number=1,
  )

  The Examples with spans > start_span_number = 1, and within the range
  [10 - 1 - 3, 10 - 1] = [6, 9] will be used for evaluation. Model 1, trained
  on spans [1, 2, 3] is the latest created Model trained on a span smaller than
  6 - 2 = 4.

  Therefore,

  {
      "model": [Model 1]
      "examples": [Example 6, Example 7, Example 8, Example 9]
  }

  will be returned.

  A SkipSignal will be present in the compiled IR if:
    1. There are no input Examples and/or Models.
    2. The evaluation span range contains spans not present in the passed in
       Examples.
    3. No Model was found that was trained on an Example with a small enough
       span.

  This means the Evaluator component (and any downstream components) will not
  run.

  Example Usage:
    evaluator_inputs = span_driven_evaluator_inputs(
        example_gen.outputs["example"], trainer.outputs["model"])
    evaluator = Evaluator(
        model=evaluator_inputs["model"],
        examples=evaluator_inputs["examples"],
        ...
    )

  Args:
    examples: The list of Examples artifacts to consider.
    models: The list of Model artifacts to consider.
    wait_spans_before_eval: If 0, as soon as a new example span appears, we
      perform an evaluation on it.  Otherwise, when example span (D +
      wait_spans_before_eval) first appears, we perform an evaluation on example
      span D.
    evaluation_training_offset: Indicates how to choose the model checkpoint for
      evaluation. When evaluating examples from span D, we choose the latest
      model checkpoint which did not train on span (D -
      evaluation_training_offset).  Note that negative values are allowed here
      intentionally.
    additional_spans_per_eval: Indicates how many spans should be included with
      an evaluation at span D. If 0, then only span D's Examples are evaluated.
      Otherwise when evaluating using span D, we will also include examples in D
      - 1, D - 2, ... up through D - additional_spans_per_eval.
    start_span_number: Indicates the smallest span number to be considered. Only
      spans >= start_span_number will be considered.

  Returns:
    A dictionary containing a single Model and the list of Examples to evaluate
    it with. The Model will be latest created eligible model.
  """
  input_dict = {
      ops_utils.EXAMPLES_KEY: examples,
      ops_utils.MODEL_KEY: models,
  }
  return ops.SpanDrivenEvaluatorInputs(
      input_dict,
      wait_spans_before_eval=wait_spans_before_eval,
      evaluation_training_offset=evaluation_training_offset,
      additional_spans_per_eval=additional_spans_per_eval,
      start_span_number=start_span_number,
  )


@span_driven_evaluator_inputs.output_type_inferrer
def _infer_span_driven_evaluator_inputs_type(
    examples: channel_type.BaseChannel,
    models: channel_type.BaseChannel,
    **kwargs,  # pylint: disable=unused-argument
):
  """Output type inferrer for span_driven_evaluator_inputs."""
  return {
      ops_utils.MODEL_KEY: models.type,
      ops_utils.EXAMPLES_KEY: examples.type,
  }
