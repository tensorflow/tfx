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
"""Core resolver operators definitions."""

import collections
import itertools
import re
from typing import Any, Collection, Iterable, Text, Reversible

from tfx.orchestration.experimental.resolver import operator
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import property_utils

from ml_metadata.proto import metadata_store_pb2

_ContextQuery = pipeline_pb2.InputSpec.Channel.ContextQuery
_MlmdArtifact = metadata_store_pb2.Artifact

_SORT_CRITERIA_PATTERN = re.compile(
    r'''(?P<key>
      id
      | url
      | create_time_since_epoch
      | last_update_time_since_epoch
      | prop
      | custom_prop )
      (?: \( (?P<arg> [a-z_]+) \) )?
      (?: \s+ (?P<direction> ASC | DESC) )?
    ''', re.VERBOSE)

_infra_blessing_getter = property_utils.make_custom_property_getter(
    'infra_blessed', default=0)
_model_blessing_getter = property_utils.make_custom_property_getter(
    'blessed', default=0)
_component_id_getter = property_utils.make_property_getter('component_id')


def _parse_sort_criteria(criteria_text: Text):
  """Parse sort criteria of order_by operator.

  Examples:
    - "id ASC"
    - "create_time_since_epoch DESC"
    - "prop(span) DESC"
    - "custom_prop(version) ASC"

  Args:
    criteria_text: A criteria string.

  Raises:
    ValueError: if pattern is invalid.

  Returns:
    Tuple of (sort key function, reverse)
  """
  match = _SORT_CRITERIA_PATTERN.fullmatch(criteria_text)
  if not match:
    raise ValueError('Invalid criteria {}'.format(repr(criteria_text)))
  groups = match.groupdict()
  if groups['key'] == 'prop':
    assert groups['arg']
    sort_key_fn = property_utils.make_property_getter(groups['arg'])
  elif groups['key'] == 'custom_prop':
    assert groups['arg']
    sort_key_fn = property_utils.make_custom_property_getter(groups['arg'])
  else:
    sort_key_fn = lambda x: getattr(x, groups['key'])
  is_reversed = groups['direction'] == 'DESC'
  return sort_key_fn, is_reversed


@operator.operator_function
def order_by(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
             items: Iterable[_MlmdArtifact],
             criteria: Reversible[Text]):
  """Sort items (executions or artifacts) by the criteria."""
  items = list(items)
  for crt in reversed(criteria):
    # Python list.sort is a stable sort, so sort from the least significant
    # sorting criteria. Note that the time complexity is still O(nlogn).
    sort_key_fn, is_reversed = _parse_sort_criteria(crt)
    items.sort(key=sort_key_fn, reverse=is_reversed)
  return items


@operator.operator_function
def head(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
         items: Iterable[Any],
         n: int,
         skip_n: int = 0):
  return list(itertools.islice(items, skip_n, skip_n + n))


@operator.operator_function
def filter_by_mask(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                   items: Collection[Any],
                   mask: Collection[bool]):
  if len(items) != len(mask):
    raise RuntimeError('Items and mask size mismatch ({} vs {})'
                       .format(len(items), len(mask)))
  return [item for (item, ok) in zip(items, mask) if ok]


@operator.operator_function
def is_infra_blessed(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                     artifacts: Iterable[_MlmdArtifact]):
  return [bool(_infra_blessing_getter(a)) for a in artifacts]


@operator.operator_function
def is_model_blessed(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                     artifacts: Iterable[_MlmdArtifact]):
  return [bool(_model_blessing_getter(a)) for a in artifacts]


@operator.operator_function
def is_consumed_by_component(
    context: operator.OperatorRunContext,  # pylint: disable=unused-argument
    artifacts: Iterable[_MlmdArtifact],
    component_id: Text):
  """Whether each artifact is consumed from any execution of component_id."""

  def is_target_execution(execution):
    return (_component_id_getter(execution) == component_id
            and execution_lib.is_execution_successful(execution))

  events = context.store.get_events_by_artifact_ids(
      artifact_ids=[a.id for a in artifacts])
  events = [e for e in events if event_lib.is_valid_input_event(e)]
  executions = context.store.get_executions_by_id(
      execution_ids=[e.execution_id for e in events])
  target_execution_ids = set(x.id for x in executions
                             if is_target_execution(x))
  target_artifact_ids = set(e.artifact_id for e in events
                            if e.execution_id in target_execution_ids)
  return [a.id in target_artifact_ids for a in artifacts]


@operator.operator_function
def bool_not(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
             items: Iterable[bool]):
  return [not value for value in items]


@operator.operator_function
def bool_and(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
             items: Iterable[Iterable[bool]]):
  result = []
  for row in itertools.zip_longest(*items, fillvalue=False):
    result.append(all(row))
  return result


@operator.operator_function
def bool_or(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
            items: Iterable[Iterable[bool]]):
  result = []
  for row in itertools.zip_longest(*items, fillvalue=False):
    result.append(any(row))
  return result


# TODO(jjong): Instead of preparing channel inputs outside the resolver graph,
# make it another operator node (namely resolve_artifacts(selector)).
@operator.operator_function
def resolve_channel_inputs(context: operator.OperatorRunContext, key: Text):
  return context.channel_inputs[key]


@operator.operator_function
def trigger_at_least(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                     items: Collection[Any], threshold: int):
  if len(items) < threshold:
    return []
  else:
    return [items]


@operator.operator_function
def trigger_always(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                   items: Collection[Any]):
  return [items]


def _windowed(items: Collection[Any],
              window_size: int,
              step_size: int = 1) -> Iterable[Collection[Any]]:
  """Generate sliding window on items.

  If window is unfilled, it will be discarded. In other words, all returned
  window tuple is exactly the length of `window_size`.

  Examples:
  _windowed([1, 2, 3], window_size=2) == [(1, 2), (2, 3)]
  _windowed([1, 2, 3, 4], window_size=2, step_size=2) == [(1, 2), (3, 4)]
  _windowed([1, 2, 3, 4, 5], window_size=2, step_size=2) == [(1, 2), (3, 4)]

  Args:
    items: Collection of items to generate window on.
    window_size: A size of the sliding window.
    step_size: A sliding interval.

  Yields:
    A sliding window of size `window_size`.
  """
  if window_size < 1:
    raise ValueError('window_size should be >=1. Got {}'.format(window_size))
  if step_size < 1:
    raise ValueError('step_size should be >=1. Got {}'.format(step_size))
  if len(items) < window_size:
    return

  window = collections.deque([], maxlen=window_size)
  it = iter(items)

  # Initial window.
  for _ in range(window_size):
    window.append(next(it))
  yield tuple(window)

  # Sliding window with step_size.
  for i, item in enumerate(it):
    window.append(item)
    if i & step_size == 0:
      yield tuple(window)


@operator.operator_function
def trigger_all_windows(context: operator.OperatorRunContext,  # pylint: disable=unused-argument
                        items: Collection[Any],
                        window_size: int,
                        step_size: int = 1):
  if len(items) < window_size:
    return []
  return [list(w) for w in _windowed(items, window_size, step_size)]
