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
      to the the smallest span in the artifacts.
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

    Although num_spans=5, only two unique span numbers are availble, 3 and 7,
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
def all_spans(artifacts, *, span_descending: bool = False):
  """Returns the sorted artifacts with unique spans.

  By default, all artifacts with unique spans (ties broken by latest version)
  are returned.

  Example usage:

    Consider 6 artifacts with:
      spans    = [1, 3, 3, 2, 8, 7]
      versions = [0, 0, 1, 0, 1, 2]

    all_spans(
        span_descending=False)

    will return artifacts:
      spans    = [1, 2, 3, 7, 8]
      versions = [0, 0, 1, 1, 2]

    Note that there are 2 artifacts with span 3, but only the one with the
    latest version is returned. Spans are sorted in ascending order.

    all_spans(
        span_descending=True)

    will return all the artifacts:
      spans    = [8, 7, 3, 2, 1]
      versions = [2, 1, 1, 0, 0]

    Spans are sorted in descending order.

  Args:
    artifacts: The artifacts to filter.
    span_descending: If true, then the artifacts will be sorted by span in
      descending order. Else, they will be sorted in ascending order by span.
      Note that sorting happens first by span and then by version, and that
      version is always sorted in ascending order.

  Returns:
    Sorted Artifacts with unique spans.
  """
  return ops.AllSpans(artifacts, span_descending=span_descending)


@resolver_function.resolver_function
def latest_pipeline_run_outputs(pipeline):
  """Returns the artifacts in the latest COMPLETE pipeline run.

  Example usage:

    producer_pipeline = Pipeline()

    consumer_pipeline_inputs = PipelineInputs(
      latest_pipeline_run_outputs(producer_pipeline))
    trainer = TFTrainer(
      examples=pipeline_inputs.inputs['examples'],
      schema=pipeline_inputs.inputs['schema'])
    consumer_pipeline = Pipeline(
        inputs=consumer_pipeline_inputs,
        components=[trainer],
    )

  Args:
    pipeline: The pipeline producing the artifacts

  Returns:
    The artifacts in the latest COMPLETE pipeline run.
  """
  return ops.LatestPipelineRunOutputs(pipeline_name=pipeline.pipeline_name)


@latest_pipeline_run_outputs.output_type_inferrer
def _infer_latest_pipeline_run_type(pipeline):
  return {
      output_key: channel.type
      for output_key, channel in pipeline.outputs.items()
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
      trainer = Trainer(examples=examples_window)

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

  resolved_artifacts = ops.SlidingWindow(
      resolved_artifacts, window_size=num_spans)

  return resolved_artifacts


@sequential_rolling_range.output_type_inferrer
def _infer_seqential_rolling_range_type(channel, **kwargs):  # pylint: disable=unused-argument
  return {'window': channel.type}
