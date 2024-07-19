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

from typing import Optional, Sequence, Union

from absl import logging
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution.ops import ops
from tfx.types import artifact
from tfx.types import channel as channel_types


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
def static_range(
    artifacts,
    *,
    start_span_number: int = -1,
    end_span_number: int = -1,
    keep_all_versions: bool = False,
    exclude_span_numbers: Sequence[int] = (),
    min_spans: Optional[int] = None,
):
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
      min_spans is None, it is set to -1, meaning all unique spans will be
      considered.

  Returns:
    Artifacts with spans in [start_span, end_span] inclusive.
  """
  resolved_artifacts = ops.StaticSpanRange(
      artifacts,
      start_span=start_span_number,
      end_span=end_span_number,
      keep_all_versions=keep_all_versions,
  )
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers
    )

  if min_spans is None:
    # We check that start_span_number and end_span_number are positive to ensure
    # min_spans is well defined. Else, it is set to -1, meaning all the unique
    # spans will be considered.
    if start_span_number >= 0 and end_span_number >= 0:
      min_spans = end_span_number - start_span_number + 1

      # Decrement min_spans by the number of spans in exclude_span_numbers that
      # are in the range [start_span_number, end_span_number].
      num_excluded_spans = 0
      for excluded_span in exclude_span_numbers:
        if (
            excluded_span >= start_span_number
            and excluded_span <= end_span_number
        ):
          num_excluded_spans += 1
      min_spans -= num_excluded_spans

      logging.warning(
          'min_spans for static_range(...) was not set and is being set to '
          'end_span_number - start_span_number + 1 - '
          '(number of excluded spans in the range [start_span, end_span]) = '
          '%s - %s + 1 - %s = %s.',
          end_span_number,
          start_span_number,
          num_excluded_spans,
          min_spans,
      )
    else:
      min_spans = -1
      logging.warning(
          'min_spans for static_range(...) was not set and is being set to -1, '
          'meaning static_range(...) will never throw a SkipSignal.'
      )

  return ops.SkipIfLessThanNSpans(resolved_artifacts, n=min_spans)


@resolver_function.resolver_function
def rolling_range(
    artifacts,
    *,
    start_span_number: int = 0,
    num_spans: int = 1,
    skip_num_recent_spans: int = 0,
    keep_all_versions: bool = False,
    exclude_span_numbers: Sequence[int] = (),
    min_spans: Optional[int] = None,
    version_sort_keys: Sequence[str] = (),
):
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
     to lack of inputs. To avoid this, you would have to decrease min_spans.

  Pythonically, this range is equivalent to:
  sorted_spans[:-skip_num_recent_spans][-num_spans:]

  This resolver function is based on the span-version semantics, which only
  considers the latest version of each span. The version semantics can be
  optionally changed by providing a list of artifact attributes that can be used
  to sort versions within a particular span. If you want to keep all versions,
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
    version_sort_keys: List of string artifact attributes to sort or filter the
      versions witin the spans, applied in order of specification. Nested keys
      can use '.' separator for e.g. 'mlmd_artifact.create_time_since_epoch'. It
      can be used to override the default behavior, which is sort by version
      number and break ties by create time and id.

  Returns:
    Artifacts with spans in the rolling range.
  """
  resolved_artifacts = ops.LatestSpan(
      artifacts,
      min_span=start_span_number,
      n=num_spans,
      skip_last_n=skip_num_recent_spans,
      keep_all_versions=keep_all_versions,
      version_sort_keys=version_sort_keys,
  )
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers
    )

  if min_spans is None:
    logging.warning(
        'min_spans for rolling_range(...) was not set, so it is defaulting to '
        'num_spans = %s. If skip_num_recent_spans is set, this may delay '
        'the component triggering on the first run until sufficient Examples '
        'artifacts are available.',
        num_spans,
    )
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
        examples=consumer_pipeline_inputs.inputs['examples'],
        schema=consumer_pipeline_inputs.inputs['schema'])
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
          f'Available: {list(pipeline.outputs)}'
      )
  return ops.LatestPipelineRunOutputs(
      pipeline_name=pipeline.pipeline_name, output_keys=output_keys
  )


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
def sequential_rolling_range(
    artifacts,
    *,
    start_span_number: Optional[int] = None,
    num_spans: int = 1,
    skip_num_recent_spans: int = 0,
    keep_all_versions: bool = False,
    exclude_span_numbers: Sequence[int] = (),
    stride: int = 1,
):
  """Returns artifacts with spans in a sequential rolling range.

  Sequential rolling range is a sliding window on the oldest consecutive spans.

  The consecutive spans must be in the range:
  [start_span_number, max_span - skip_num_recent_spans], where max_span is the
  maximum span present in the artifacts. This range is modified to account for
  exclude_span_numbers, for details see the ConsecutiveSpans ResolverOp
  implementation.

  The window size is num_spans and the sliding window has a default stride of 1.
  If the spans are not consecutive, then the sequential rolling range waits for
  the missing span to arrive.

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
    stride=1 applied:

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
    stride: The step size of the sliding window. Must be > 0, defaults to 1.

  Returns:
    Artifacts with spans in the sequential rolling range.
  """
  resolved_artifacts = ops.ConsecutiveSpans(
      artifacts,
      first_span=start_span_number if start_span_number is not None else -1,
      skip_last_n=skip_num_recent_spans,
      keep_all_versions=keep_all_versions,
      denylist=exclude_span_numbers,
  )

  return ops.SlidingWindow(
      resolved_artifacts, window_size=num_spans, stride=stride
  )


@sequential_rolling_range.output_type_inferrer
def _infer_seqential_rolling_range_type(channel, **kwargs):  # pylint: disable=unused-argument
  return {'window': channel.type}


@resolver_function.resolver_function()
def paired_spans(
    artifacts,
    *,
    match_version: bool = True,
    keep_all_versions: bool = False,
):
  """Pairs up Examples from different channels, matching by (span, version).

  This enables grouping together Artifacts from separate channels.

  Example usage:

  NOTE: Notation here is `{artifact_type}:{span}:{version}`

  >>> paired_spans({'x': channel([X:0:0, X:0:1, X:1:0, X:2:0]),
                    'y': channel([Y:0:0, Y:0:1, Y:1:0, Y:3:0])})
  Loopable([
      {'x': channel([X:0:1]), 'y': channel([Y:0:1])},
      {'x': channel([X:1:0]), 'y': channel([Y:1:0])},
  ])

  Note that the span `0` has two versions, but only the latest version `1` is
  selected. This is the default semantics of the span & version where only the
  latest version is considered valid of each span.

  If you want to select all versions including the non-latest ones, you can
  set `keep_all_versions=True`.

  >>> paired_spans({'x': channel([X:0:0, X:0:1, X:1:0]),
                    'y': channel([Y:0:0, Y:0:1, Y:1:0]},
                    keep_all_versions=True)
  Loopable([
      {'x': channel([X:0:0]), 'y': channel([Y:0:0])},
      {'x': channel([X:0:1]), 'y': channel([Y:0:1])},
      {'x': channel([X:1:0]), 'y': channel([Y:1:0])},
  ])

  By default, the version property is considered for pairing, meaning that the
  version should exact match, otherwise it is not considered the pair.

  >>> paired_spans({'x': channel([X:0:999, X:1:999]),
                    'y': channel([Y:0:0, Y:1:0])})
  Loopable([])

  If you do not care about version, and just want to pair artifacts that
  consider only the span property (and select latest version for each span),
  you can set `match_version=False`.

  >>> paired_spans({'x': channel([X:0:999, X:1:999]),
                    'y': channel([Y:0:0, Y:1:0, Y:1:1])},
                    match_version=False)
  Loopable([
      {'x': channel([X:0:999]), 'y': channel([Y:0:0])},
      {'x': channel([X:1:999]), 'y': channel([Y:1:1])},
  ])

  Since `match_version=False` only consideres the latest version of each span,
  this cannot be used together with `keep_all_versions=True`.

  As `paired_spans` returns a `Loopable`, it must be used together with
  `ForEach`. For example:

  ```python
  with ForEach(paired_spans({'a' : channel_a, 'b' : channel_b})) as pair:
    component = Component(a=pair['a'], b=pair['b'])
  ```

  NOTE: `paired_spans` can pair Artifacts from N >= 2 channels.

  Args:
    artifacts: A dictionary of artifacts.
    match_version: Whether the version of each span should exactly match.
    keep_all_versions: Whether to pair up all versions of artifacts, or only the
      latest version. Defaults to False. Requires match_version = True.

  Returns:
    A list of artifact dicts where each dict has as its key the channel key,
    and as its value has a list with a single artifact having the same span and
    version across the dict.
  """
  if keep_all_versions and not match_version:
    raise ValueError('keep_all_versions = True requires match_version = True.')

  # TODO: b/322812375 - Remove kwargs dict handling once orchestrator knows
  # match_version argument.
  kwargs = {}
  if not match_version:
    kwargs['match_version'] = False
  return ops.PairedSpans(
      artifacts,
      keep_all_versions=keep_all_versions,
      **kwargs,
  )


@resolver_function.resolver_function
def filter_property_equal(
    artifacts,
    *,
    key: str,
    value: Union[int, float, str, bool, artifact.JsonValueType],
):
  """Returns artifacts with matching property values.

  Example usage:

  Consider artifacts [A, B, C] with bool property 'blessed' set to
  [True, True, False].

  filter_property_equal(
      [A, B, C],
      key='blessed',
      value=False,
  )

  will return [C].

  Args:
    artifacts: The list of artifacts to filter.
    key: The property key to match by.
    value: The expected property value to match by.

  Returns:
    Artifact(s) with matching custom property (or property) values.
  """
  return ops.EqualPropertyValues(
      artifacts,
      property_key=key,
      property_value=value,
      is_custom_property=False,
  )


@filter_property_equal.output_type_inferrer
def _infer_filter_property_equal_type(
    channel: channel_types.BaseChannel, **kwargs  # pylint: disable=unused-argument
):
  return channel.type


@resolver_function.resolver_function
def filter_custom_property_equal(
    artifacts,
    *,
    key: str,
    value: Union[int, float, str, bool, artifact.JsonValueType],
):
  """Returns artifacts with matching custom property values.

  Example usage:

  Consider artifact [A, B, C] with int custom property 'purity' set to
  [1, 1, 2].

  filter_custom_property_equal(
      [A, B, C],
      key='purity',
      value=2,
  )

  will return [C].

  Args:
    artifacts: The list of artifacts to filter.
    key: The property key to match by.
    value: The expected property value to match by.

  Returns:
    Artifact(s) with matching custom property (or property) values.
  """
  return ops.EqualPropertyValues(
      artifacts,
      property_key=key,
      property_value=value,
      is_custom_property=True,
  )


@filter_custom_property_equal.output_type_inferrer
def _infer_filter_custom_property_equal_type(
    channel: channel_types.BaseChannel, **kwargs  # pylint: disable=unused-argument
):
  return channel.type


@resolver_function.resolver_function
def _slice(artifacts, **kwargs):
  # It's important to not pass the None value which cannot be serialized to IR.
  kwargs = {k: v for k, v in kwargs.items() if v is not None}
  return ops.Slice(artifacts, **kwargs)


def pick(channel: channel_types.BaseChannel, i: int, /):
  """Pick an i'th artifact from channel.

  Like in python, negative indexing is allowed.

  If the index is out of range, in synchronous pipeline it raises an error and
  the component would not get executed. In asynchronous pipeline, it will wait
  until the input length is sufficient to handle the index.

  Usage:

  ```python
  # In ASYNC pipeline:
  with ForEach(example_gen.outputs['examples']) as each_example:
    statistics_gen = StatisticsGen(examples=each_example)

  latest_statistics_pair = latest_created(
      statistics_gen.outputs['statistics'], n=2
  )
  validator = DistributionValidator(
      baseline_statistics=pick(latest_statistics_pair, 0),
      statistics=pick(latest_statistics_pair, 1),
      ...
  )
  ```

  Args:
    channel: A channel instance (e.g. `my_component.outputs['x']`).
    i: An index to pick. Can be negative.

  Returns:
    A channel that represents `inputs[i]`.
  """
  return _slice(channel, start=i, stop=(i + 1) or None, min_count=1)


def slice(  # pylint: disable=redefined-builtin
    channel: channel_types.BaseChannel,
    /,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    min_count: Optional[int] = None,
):
  """Pick slice(start, stop) of the input artifacts.

  Like in python, negative indexing is allowed.

  If the range is larger than the number of artifacts in the channel, then like
  in python slice, the result would be truncated to the available values. You
  can use min_count to ensure the range has enough values. In synchronous
  pipeline, it is an error if the min_count is not met. In asynchronous
  pipeline, it will wait until min_count is met.

  None value in the start or stop index means beginning and the end of the range
  respectively. For example, `pick_range(x, start=-2, end=None)` means `x[-2:]`.

  Usage:

  ```python
  # In asynchronous pipeline
  last_week_before_yesterday = pick_range(
      example_gen.outputs['examples'], start=-7, stop=-1
  )
  with ForEach(last_week_before_yesterday) as each_example:
    evaluator = Evaluator(example=each_example, model=latest_model)
  ```

  Args:
    channel: A channel instance (e.g. `my_component.outputs['x']`).
    start: A start index (inclusive) of the range. Can be negative.
    stop: A stop index (exclusive) of the range. Can be negative.
    min_count: A minimum number of values the range should contain. When
      specified, synchronous (DAG) pipeline will fail if the min_count is not
      met. Asynchronous (continuous) pipeine will wait until min_count is met.

  Returns:
    A channel that represents `inputs[start:stop]` slice range.
  """
  # Slice(start=None, stop=None) is a no-op and we can return the input as is.
  if start is None and stop is None:
    return channel
  return _slice(channel, start=start, stop=stop, min_count=min_count)


@resolver_function.resolver_function(unwrap_dict_key='window')
def sliding_window(channel: channel_types.BaseChannel, window_size: int):
  """Returns artifacts with a sliding window applied.

  For example, for a channel with artifacts [A, B, C, D] and window_size = 2,
  [[A, B], [B, C], [C, D]] will be returned.

  For Examples artifacts, sequential_rolling_range() should be used instead.

  Because sliding_window() returns multiple windows, it must be used
  together with ForEach.

  Usage:

  ```python
  # In ASYNC pipeline:
  with ForEach(
      sliding_window(statistics_gen.outputs['statistics'], window_size=2)
  ) as statistics_pair:
    distribution_validator = DistributionValidator(
        baseline_statistics=pick(statistics_pair, 0),
        statistics=pick(statistics_pair, 1),
        ...
    )
  ```

  Args:
    channel: A channel instance (e.g. `my_component.outputs['x']`).
    window_size: The length of the sliding window, must be > 0.

  Returns:
    Artifacts with a sliding window applied.
  """
  return ops.SlidingWindow(channel, window_size=window_size)


@sliding_window.output_type_inferrer
def _infer_sliding_window_type(channel: channel_types.BaseChannel, **kwargs):  # pylint: disable=unused-argument
  return {'window': channel.type}
