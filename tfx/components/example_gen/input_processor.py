# Copyright 2021 Google LLC. All Rights Reserved.
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
"""InputProcessor for resolving span and version."""

import abc
import copy
from typing import Iterable, Optional, Text, Tuple
import six

from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


class InputProcessor(six.with_metaclass(abc.ABCMeta, object)):
  """Base InputProcessor class."""

  def __init__(self,
               splits: Iterable[example_gen_pb2.Input.Split],
               range_config: Optional[range_config_pb2.RangeConfig] = None):
    """Initialize InputProcessor.

    Args:
      splits: An iterable collection of example_gen_pb2.Input.Split objects.
      range_config: An instance of range_config_pb2.RangeConfig, defines the
        rules for span resolving.
    """
    self._splits = splits

    if range_config:
      if range_config.HasField('static_range'):
        start_span_number = range_config.static_range.start_span_number
        end_span_number = range_config.static_range.end_span_number
        if start_span_number != end_span_number:
          raise ValueError(
              'For ExampleGen, start and end span numbers for RangeConfig.StaticRange must be equal.'
          )
      elif range_config.HasField('rolling_range'):
        if range_config.rolling_range.num_spans != 1:
          raise ValueError(
              'ExampleGen only support single span for RangeConfig.RollingRange.'
          )
        if range_config.rolling_range.start_span_number > 0:
          raise ValueError(
              'RangeConfig.rolling_range.start_span_number is not supported.')
      else:
        raise ValueError('Only static_range and rolling_range are supported.')

    self._range_config = range_config

  # TODO(b/181275944): consider move this out of input_processor.
  def resolve_span_and_version(self) -> Tuple[int, Optional[int]]:
    """Resolves Span and Version information.

    If a pattern has the {SPAN} placeholder or the Date spec placeholders,
    {YYYY}, {MM}, and {DD}, and optionally, the {VERSION} placeholder, attempts
    to find aligned values that results in all splits having the target span and
    most recent version for that span.

    Returns:
      A Tuple of [target_span, target_version], where:
      1. target_span is either the value matched with the {SPAN} placeholder,
         the value mapped from matching the calendar date with the date
         placeholders {YYYY}, {MM}, {DD} or 0 if a placeholder wasn't specified.
      2. target_version is either the value matched with the {VERSION}
         placeholder, or None if the placeholder wasn't specified.
    """
    target_span = 0
    target_version = None

    if self._range_config:
      if self._range_config.HasField('static_range'):
        target_span = self._range_config.static_range.start_span_number
      elif self._range_config.HasField('rolling_range'):
        target_span = self.get_latest_span()

    # TODO(b/162622803): add default behavior for when version spec not present.
    target_version = self.get_latest_version(target_span)  # pylint: disable=assignment-from-none

    return target_span, target_version

  @abc.abstractmethod
  def get_pattern_for_span_version(self, pattern: Text, span: int,
                                   version: Optional[int]) -> Text:
    """Return pattern with Span and Version spec filled."""
    # TODO(b/181275944): refactor as not all type of ExampleGen has pattern.
    raise NotImplementedError

  @abc.abstractmethod
  def get_latest_span(self) -> int:
    """Resolves the latest Span information."""
    raise NotImplementedError

  def get_latest_version(self, span: int) -> Optional[int]:
    """Resolves the latest Version of a Span."""
    return None

  def get_input_fingerprint(self, span: int,
                            version: Optional[int]) -> Optional[Text]:
    """Returns the fingerprint for a certain Version of a certain Span."""
    return None


class FileBasedInputProcessor(InputProcessor):
  """Custom InputProcessor for file based ExampleGen driver."""

  def __init__(self,
               input_base_uri: Text,
               splits: Iterable[example_gen_pb2.Input.Split],
               range_config: Optional[range_config_pb2.RangeConfig] = None):
    """Initialize FileBasedInputProcessor.

    Args:
      input_base_uri: The base path from which files will be searched.
      splits: An iterable collection of example_gen_pb2.Input.Split objects.
      range_config: An instance of range_config_pb2.RangeConfig, defines the
        rules for span resolving.
    """
    super(FileBasedInputProcessor, self).__init__(
        splits=splits, range_config=range_config)

    self._is_match_span = None
    self._is_match_date = None
    self._is_match_version = None
    for split in splits:
      is_match_span, is_match_date, is_match_version = utils.verify_split_pattern_specs(
          split)
      if self._is_match_span is None:
        self._is_match_span = is_match_span
        self._is_match_date = is_match_date
        self._is_match_version = is_match_version
      elif (self._is_match_span != is_match_span or
            self._is_match_date != is_match_date or
            self._is_match_version != is_match_version):
        raise ValueError('Spec setup should the same for all splits: %s.' %
                         split.pattern)

    if (self._is_match_span or self._is_match_date) and not range_config:
      range_config = range_config_pb2.RangeConfig(
          rolling_range=range_config_pb2.RollingRange(num_spans=1))
    if not self._is_match_span and not self._is_match_date and range_config:
      raise ValueError(
          'Span or Date spec should be specified in split pattern if RangeConfig is specified.'
      )

    self._input_base_uri = input_base_uri
    self._fingerprint = None

  def resolve_span_and_version(self) -> Tuple[int, Optional[int]]:
    # TODO(b/181275944): refactor to use base resolve_span_and_version.
    splits = []
    for split in self._splits:
      splits.append(copy.deepcopy(split))
    self._fingerprint, span, version = utils.calculate_splits_fingerprint_span_and_version(
        self._input_base_uri, splits, self._range_config)
    return span, version

  def get_pattern_for_span_version(self, pattern: Text, span: int,
                                   version: Optional[int]) -> Text:
    """Return pattern with Span and Version spec filled."""
    return utils.get_pattern_for_span_version(
        pattern=pattern,
        is_match_span=self._is_match_span,
        is_match_date=self._is_match_date,
        is_match_version=self._is_match_version,
        span=span,
        version=version)

  def get_latest_span(self) -> int:
    """Resolves the latest Span information."""
    raise NotImplementedError

  def get_latest_version(self, span: int) -> Optional[int]:
    """Resolves the latest Version of a Span."""
    raise NotImplementedError

  def get_input_fingerprint(self, span: int,
                            version: Optional[int]) -> Optional[Text]:
    """Returns the fingerprint for a certain Version of a certain Span."""
    assert self._fingerprint, 'Call resolve_span_and_version first.'
    return self._fingerprint


class QueryBasedInputProcessor(InputProcessor):
  """Custom InputProcessor for query based ExampleGen driver."""

  def get_pattern_for_span_version(self, pattern: Text, span: int,
                                   version: Optional[int]) -> Text:
    """Return pattern with Span and Version spec filled."""
    return utils.get_query_for_span(pattern, span)

  def get_latest_span(self) -> int:
    """Resolves the latest Span information."""
    # TODO(b/179853017): support latest span based on timestamp.
    raise NotImplementedError(
        'For QueryBasedExampleGen, latest Span is not supported, please use RangeConfig.StaticRange.'
    )

  def get_latest_version(self, span: int) -> Optional[int]:
    """Resolves the latest Version of a Span."""
    # TODO(b/179853017): support Version.
    return None

  def get_input_fingerprint(self, span: int,
                            version: Optional[int]) -> Optional[Text]:
    """Returns the fingerprint for a certain Version of a certain Span."""
    # TODO(b/179853017): support fingerprint of table.
    return None
