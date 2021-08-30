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
"""Utilities for ExampleGen components."""

import datetime
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from tfx.dsl.io import fileio
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.utils import io_utils
from google.protobuf import json_format


# Key for the `payload_format` custom property of output examples artifact.
PAYLOAD_FORMAT_PROPERTY_NAME = 'payload_format'
# Key for the `file_format` custom property of output examples artifact.
FILE_FORMAT_PROPERTY_NAME = 'file_format'
# Key for the `input_fingerprint` custom property of output examples artifact.
FINGERPRINT_PROPERTY_NAME = 'input_fingerprint'
# Key for the `span` custom property of output examples artifact.
SPAN_PROPERTY_NAME = 'span'
# Span spec used in split pattern.
SPAN_SPEC = '{SPAN}'
# Span spec regex to capture width modifier. This matches the spec '{SPAN:x}'
# and captures the string 'x'.
SPAN_SPEC_WIDTH_REGEX = '{SPAN:(?P<width>.*?)}'
# Full regex for matching span specs with or without width modifier.
SPAN_FULL_REGEX = '{}|{}'.format(SPAN_SPEC, SPAN_SPEC_WIDTH_REGEX)
# Key for the `version` custom property of output examples artifact.
VERSION_PROPERTY_NAME = 'version'
# Version spec used in split pattern.
VERSION_SPEC = '{VERSION}'
# Version spec regex to capture width modifier. This matches the spec
# '{VERSION:x}' and captures the string 'x'.
VERSION_SPEC_WIDTH_REGEX = '{VERSION:(?P<width>.*?)}'
# Full regex for matching version specs with or without width modifier.
VERSION_FULL_REGEX = '{}|{}'.format(VERSION_SPEC, VERSION_SPEC_WIDTH_REGEX)
# Date specs used in split pattern.
YEAR_SPEC = '{YYYY}'
MONTH_SPEC = '{MM}'
DAY_SPEC = '{DD}'
# Order of importance for Date specs.
DATE_SPECS = [YEAR_SPEC, MONTH_SPEC, DAY_SPEC]
# Specs for query:
#   @span_begin_timestamp: Start of span interval, Timestamp in seconds.
#   @span_end_timestamp: End of span interval, Timestamp in seconds.
#   @span_yyyymmdd_utc: STRING with format, e.g., '20180114', corresponding to
#                       the span interval begin in UTC.
#   Query examples,
#     1. SELECT * FROM table WHERE date = @span_yyyymmdd_utc
#     2. SELECT * FROM table WHERE timestamp
#        BETWEEN @span_begin_timestamp AND @span_end_timestamp
SPAN_BEGIN_TIMESTAMP = '@span_begin_timestamp'
SPAN_END_TIMESTMAP = '@span_end_timestamp'
SPAN_YYYYMMDD_UTC = '@span_yyyymmdd_utc'
# Unix epoch date to calculate span number from.
UNIX_EPOCH_DATE = datetime.datetime(1970, 1, 1)
UNIX_EPOCH_DATE_UTC = datetime.datetime(  # pylint: disable=g-tzinfo-datetime
    1970,
    1,
    1,
    tzinfo=datetime.timezone.utc)

_DEFAULT_ENCODING = 'utf-8'


def dict_to_example(instance: Dict[str, Any]) -> tf.train.Example:
  """Converts dict to tf example."""
  feature = {}
  for key, value in instance.items():
    # TODO(jyzhao): support more types.
    if value is None:
      feature[key] = tf.train.Feature()
    elif isinstance(value, int):
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif isinstance(value, str):
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[value.encode(_DEFAULT_ENCODING)]))
    elif isinstance(value, list):
      if not value:
        feature[key] = tf.train.Feature()
      elif isinstance(value[0], int):
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
      elif isinstance(value[0], float):
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=value))
      elif isinstance(value[0], str):
        feature[key] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[v.encode(_DEFAULT_ENCODING) for v in value]))
      else:
        raise RuntimeError('Column type `list of {}` is not supported.'.format(
            type(value[0])))
    else:
      raise RuntimeError('Column type {} is not supported.'.format(type(value)))
  return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_output_split_names(
    input_config: Union[example_gen_pb2.Input, Dict[str, Any]],
    output_config: Union[example_gen_pb2.Output, Dict[str, Any]]) -> List[str]:
  """Return output split name based on input and output config.

  Return output split name if it's specified and input only contains one split,
  otherwise output split will be same as input.

  Args:
    input_config: example_gen_pb2.Input instance. If any field is provided as a
      RuntimeParameter, input_config should be constructed as a dict with the
      same field names as Input proto message.
    output_config: example_gen_pb2.Output instance. If any field is provided as
      a RuntimeParameter, output_config should be constructed as a dict with the
      same field names as Output proto message.

  Returns:
    List of split names.

  Raises:
    RuntimeError: if configs are not valid, including:
      - Missing field.
      - Duplicated split.
      - Output split is specified while input has more than one split.
      - Missing train and eval split.
  """
  result = []
  # Convert proto to dict for easy sanity check. Otherwise we need to branch the
  # logic based on parameter types.
  if isinstance(output_config, example_gen_pb2.Output):
    output_config = json_format.MessageToDict(
        output_config,
        including_default_value_fields=True,
        preserving_proto_field_name=True)
  if isinstance(input_config, example_gen_pb2.Input):
    input_config = json_format.MessageToDict(
        input_config,
        including_default_value_fields=True,
        preserving_proto_field_name=True)

  if 'split_config' in output_config and 'splits' in output_config[
      'split_config']:
    if 'splits' not in input_config:
      raise RuntimeError(
          'ExampleGen instance specified output splits but no input split '
          'is specified.')
    if len(input_config['splits']) != 1:
      # If output is specified, then there should only be one input split.
      raise RuntimeError(
          'ExampleGen instance specified output splits but at the same time '
          'input has more than one split.')
    for split in output_config['split_config']['splits']:
      if not split['name'] or (isinstance(split['hash_buckets'], int) and
                               split['hash_buckets'] <= 0):
        raise RuntimeError('Str-typed output split name and int-typed '
                           'hash buckets are required.')
      result.append(split['name'])
  else:
    # If output is not specified, it will have the same split as the input.
    if 'splits' in input_config:
      for split in input_config['splits']:
        if not split['name'] or not split['pattern']:
          raise RuntimeError('Str-typed input split name and pattern '
                             'are required.')
        result.append(split['name'])

  if not result:
    raise RuntimeError('ExampleGen splits are missing.')
  if len(result) != len(set(result)):
    raise RuntimeError('Duplicated split name {}.'.format(result))

  return result


def make_default_input_config(
    split_pattern: str = '*') -> example_gen_pb2.Input:
  """Returns default input config."""
  # Treats input base dir as a single split.
  return example_gen_pb2.Input(splits=[
      example_gen_pb2.Input.Split(name='single_split', pattern=split_pattern)
  ])


def make_default_output_config(
    input_config: Union[example_gen_pb2.Input, Dict[str, Any]]
) -> example_gen_pb2.Output:
  """Returns default output config based on input config."""
  if isinstance(input_config, example_gen_pb2.Input):
    input_config = json_format.MessageToDict(
        input_config,
        including_default_value_fields=True,
        preserving_proto_field_name=True)

  if len(input_config['splits']) > 1:
    # Returns empty output split config as output split will be same as input.
    return example_gen_pb2.Output()
  else:
    # Returns 'train' and 'eval' splits with size 2:1.
    return example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))


def _glob_to_regex(glob_pattern: str) -> str:
  """Changes glob pattern to regex pattern."""
  regex_pattern = glob_pattern
  regex_pattern = regex_pattern.replace('.', '\\.')
  regex_pattern = regex_pattern.replace('+', '\\+')
  regex_pattern = regex_pattern.replace('*', '[^/]*')
  regex_pattern = regex_pattern.replace('?', '[^/]')
  regex_pattern = regex_pattern.replace('(', '\\(')
  regex_pattern = regex_pattern.replace(')', '\\)')
  return regex_pattern


def date_to_span_number(year: int, month: int, day: int) -> int:
  """Calculated span from date as number of days since unix epoch."""
  return (datetime.datetime(year, month, day) - UNIX_EPOCH_DATE).days


def span_number_to_date(span: int) -> Tuple[int, int, int]:
  """Given a span number, convert it to the corresponding calendar date."""
  date = UNIX_EPOCH_DATE + datetime.timedelta(span)
  return date.year, date.month, date.day


def _make_zero_padding_spec_value(spec_full_regex: str, pattern: str,
                                  spec_value: int) -> str:
  """Returns spec value, applies zero padding if needed."""
  match_result = re.search(spec_full_regex, pattern)
  assert match_result, 'No %s found in split %s' % (spec_full_regex, pattern)
  width_str = match_result.group('width')
  if width_str:
    width_int = 0
    try:
      width_int = int(width_str)
    except ValueError:
      raise ValueError('Width modifier is not a integer: %s' % pattern)
    if width_int <= 0:
      raise ValueError('Width modifier is not positive: %s' % pattern)
    if width_int < len(str(spec_value)):
      raise ValueError(
          'Spec width is less than number of digits in spec: (%s, %s)' %
          (width_int, spec_value))
    return str(spec_value).zfill(width_int)
  return str(spec_value)


def verify_split_pattern_specs(
    split: example_gen_pb2.Input.Split) -> Tuple[bool, bool, bool]:
  """Verify and identify specs to be matched in split pattern."""
  # Match occurences of pattern '{SPAN}|{SPAN:*}'. If it exists, capture
  # span width modifier. Otherwise, the empty string is captured.
  span_matches = re.findall(SPAN_FULL_REGEX, split.pattern)
  is_match_span = bool(span_matches)

  # Match occurences of pattern '{VERSION}|{VERSION:*}'. If it exists, capture
  # version width modifier. Otherwise, the empty string is captured.
  version_matches = re.findall(VERSION_FULL_REGEX, split.pattern)
  is_match_version = bool(version_matches)

  is_match_date = any(spec in split.pattern for spec in DATE_SPECS)

  if [is_match_span, is_match_date].count(True) > 1:
    raise ValueError(
        'Either span spec or date specs must be specified exclusively in %s' %
        split.pattern)

  if is_match_span and len(span_matches) != 1:
    raise ValueError('Only one %s is allowed in %s' %
                     (SPAN_SPEC, split.pattern))

  if is_match_date and not all(
      split.pattern.count(spec) == 1 for spec in DATE_SPECS):
    raise ValueError(
        'Exactly one of each date spec (%s, %s, %s) is required in %s' %
        (YEAR_SPEC, MONTH_SPEC, DAY_SPEC, split.pattern))

  if is_match_version and (not is_match_span and not is_match_date):
    raise ValueError(
        'Version spec provided, but Span or Date spec is not present in %s' %
        split.pattern)

  if is_match_version and len(version_matches) != 1:
    raise ValueError('Only one %s is allowed in %s' %
                     (VERSION_SPEC, split.pattern))

  return is_match_span, is_match_date, is_match_version


def get_pattern_for_span_version(pattern: str, is_match_span: bool,
                                 is_match_date: bool, is_match_version: bool,
                                 span: int, version: Optional[int]) -> str:
  """Return pattern with Span and Version spec filled."""
  if is_match_span:
    span_token = _make_zero_padding_spec_value(SPAN_FULL_REGEX, pattern, span)
    pattern = re.sub(SPAN_FULL_REGEX, span_token, pattern)
  elif is_match_date:
    year, month, day = span_number_to_date(span)
    date_tokens = [str(year).zfill(4), str(month).zfill(2), str(day).zfill(2)]
    for spec, value in zip(DATE_SPECS, date_tokens):
      pattern = pattern.replace(spec, value)
  if is_match_version:
    version_token = _make_zero_padding_spec_value(VERSION_FULL_REGEX, pattern,
                                                  version)
    pattern = re.sub(VERSION_FULL_REGEX, version_token, pattern)

  return pattern


def get_query_for_span(pattern: str, span: int) -> str:
  """Return query with timestamp placeholders filled."""
  # TODO(b/179853017): make UNIX_EPOCH_DATE_UTC timezone configurable.
  begin = UNIX_EPOCH_DATE_UTC + datetime.timedelta(days=span)
  end = begin + datetime.timedelta(days=1)
  pattern = pattern.replace(SPAN_BEGIN_TIMESTAMP, str(int(begin.timestamp())))
  pattern = pattern.replace(SPAN_END_TIMESTMAP, str(int(end.timestamp())))
  pattern = pattern.replace(
      SPAN_YYYYMMDD_UTC,
      begin.astimezone(datetime.timezone.utc).strftime("'%Y%m%d'"))
  return pattern


def _find_matched_span_version_from_path(
    file_path: str, split_regex_pattern: str, is_match_span: bool,
    is_match_date: bool, is_match_version: bool
) -> Tuple[Optional[List[str]], Optional[int], Optional[str], Optional[int]]:
  """Finds the span tokens and number given a file path and split regex."""

  result = re.search(split_regex_pattern, file_path)
  if result is None:
    raise ValueError('Glob pattern does not match regex pattern')

  matched_span_tokens = None
  matched_span_int = None
  matched_version = None
  matched_version_int = None

  if is_match_span:
    matched_span_tokens = [result.group(SPAN_PROPERTY_NAME)]
    try:
      matched_span_int = int(matched_span_tokens[0])
    except ValueError:
      raise ValueError('Cannot find %s number from %s based on %s' %
                       (SPAN_PROPERTY_NAME, file_path, split_regex_pattern))
  elif is_match_date:
    matched_span_tokens = [
        result.group(name) for name in ['year', 'month', 'day']
    ]
    try:
      matched_span_ints = [int(elem) for elem in matched_span_tokens]
    except ValueError:
      raise ValueError('Cannot find %s number using date from %s based on %s' %
                       (SPAN_PROPERTY_NAME, file_path, split_regex_pattern))
    try:
      matched_span_int = date_to_span_number(*matched_span_ints)
    except ValueError:
      raise ValueError('Retrieved date is invalid for file: %s' % file_path)

  if is_match_version:
    matched_version = result.group(VERSION_PROPERTY_NAME)
    try:
      matched_version_int = int(matched_version)
    except ValueError:
      raise ValueError('Cannot find %s number from %s based on %s' %
                       (VERSION_PROPERTY_NAME, file_path, split_regex_pattern))

  return (matched_span_tokens, matched_span_int, matched_version,
          matched_version_int)


def _get_spec_width(spec_full_regex: str, spec_name: str,
                    split: example_gen_pb2.Input.Split) -> Optional[str]:
  """Returns width modifier of a spec, if it exists."""
  result = re.search(spec_full_regex, split.pattern)
  assert result, 'No %s found in split %s' % (spec_name, split.pattern)
  width_str = result.group('width')
  if width_str:
    try:
      width_int = int(width_str)
      if width_int <= 0:
        raise ValueError('Not a positive integer.')
    except ValueError:
      raise ValueError(
          'Width modifier in %s spec is not a positive integer: %s' %
          (spec_name, split.pattern))
  return width_str


def _get_span_replace_glob_and_regex(
    range_config: range_config_pb2.RangeConfig, is_match_span: bool,
    is_match_date: bool,
    span_width_str: Optional[str]) -> Union[str, List[str]]:
  """Replace span or date spec if static range RangeConfig is provided."""
  if range_config.HasField('static_range'):
    if is_match_span:
      # If using RangeConfig.static_range, replace span spec in patterns
      # with given span from static range.
      span_str = str(range_config.static_range.start_span_number)
      if span_width_str:
        span_width_int = int(span_width_str)
        if span_width_int < len(span_str):
          raise ValueError(
              'Span spec width is less than number of digits in span: (%s, %s)'
              % (span_width_int, span_str))
        span_str = span_str.zfill(span_width_int)
      return span_str

    elif is_match_date:
      # If using RangeConfig.static_range, replace date specs in patterns
      # with calendar date derived from given span from static range.
      span_int = range_config.static_range.start_span_number
      year, month, day = span_number_to_date(span_int)
      date_tokens = [str(year).zfill(4), str(month).zfill(2), str(day).zfill(2)]
      return date_tokens

    else:
      raise ValueError('One of Span or Date should be specified.')
  else:
    raise ValueError('Only static_range in RangeConfig is supported.')


def _create_matching_glob_and_regex(
    uri: str, split: example_gen_pb2.Input.Split, is_match_span: bool,
    is_match_date: bool, is_match_version: bool,
    range_config: Optional[range_config_pb2.RangeConfig]) -> Tuple[str, str]:
  """Constructs glob and regex patterns for matching span and version.

  Construct a glob and regex pattern for matching files and capturing span and
  version information. By default, this method replaces the span, date, and
  or version specs in the split pattern with wildcard characters to get a
  glob pattern and with greedy named capture groups to get a regex pattern.

  If a static range `range_config` is specified, this method replaces the span
  spec (if `is_match_span`) in both the glob and regex pattern with the span
  number corresponding to the provided static range. If a span width modifier
  is specified, this substitution is also made with zero padding. Similarly, if
  `is_match_date`, the provided span number from the static range is converted
  is mapped back into a calendar date, which is then used to replace the date
  specs in the glob and regex patterns.

  Args:
    uri: The base path from which files will be searched.
    split: An example_gen_pb2.Input.Split object which contains a split pattern,
      to be searched on.
    is_match_span: Flag set to True if span spec is present, False otherwise.
    is_match_date: Flag set to True if date specs are present, False otherwise.
    is_match_version: Flag set to True if version spec is presen, False
      otherwise.
    range_config: An instance of range_config_pb2.RangeConfig, which specifies
      which spans to consider when finding the most recent span and version. If
      unset, search for latest span number with no restrictions.

  Returns:
    Tuple of two strings, first of which is a glob pattern to identify relevant
    files for process, the second of which is a regex pattern containing capture
    groups, for span, date, and/or version (if their respective matching flags
    are set).
  """
  split_pattern = os.path.join(uri, split.pattern)
  split_glob_pattern = split_pattern
  split_regex_pattern = _glob_to_regex(split_pattern)

  if is_match_span:
    # Check if span spec has any width args. Defaults to greedy matching if
    # no width modifiers are present.
    span_glob_replace = '*'
    span_regex_replace = '.*'

    span_width_str = _get_spec_width(SPAN_FULL_REGEX, SPAN_PROPERTY_NAME, split)
    if span_width_str:
      span_regex_replace = '.{%s}' % span_width_str

    if range_config and range_config.HasField('static_range'):
      span_str = _get_span_replace_glob_and_regex(range_config, is_match_span,
                                                  is_match_date, span_width_str)
      span_regex_replace = span_str
      span_glob_replace = span_str

    split_glob_pattern = re.sub(SPAN_FULL_REGEX, span_glob_replace,
                                split_glob_pattern)
    span_capture_regex = '(?P<{}>{})'.format(SPAN_PROPERTY_NAME,
                                             span_regex_replace)
    split_regex_pattern = re.sub(SPAN_FULL_REGEX, span_capture_regex,
                                 split_regex_pattern)

  elif is_match_date:
    date_glob_replace = ['*', '*', '*']
    # Defines a clear number of digits for certain element of date, in order of
    # year, month, and day. This covers cases where date stamps may not have
    # seperators between them.
    date_regex_replace = ['.{4}', '.{2}', '.{2}']

    if range_config and range_config.HasField('static_range'):
      date_tokens = _get_span_replace_glob_and_regex(range_config,
                                                     is_match_span,
                                                     is_match_date, None)
      date_glob_replace = date_tokens
      date_regex_replace = date_tokens

    for spec, replace in zip(DATE_SPECS, date_glob_replace):
      split_glob_pattern = split_glob_pattern.replace(spec, replace)

    for spec, replace, spec_name in zip(DATE_SPECS, date_regex_replace,
                                        ['year', 'month', 'day']):
      split_regex_pattern = split_regex_pattern.replace(
          spec, '(?P<{}>{})'.format(spec_name, replace))

  if is_match_version:
    # Check if version spec has any width modifier. Defaults to greedy matching
    # if no width modifiers are present.
    version_width_regex = '.*'

    version_width_str = _get_spec_width(VERSION_FULL_REGEX,
                                        VERSION_PROPERTY_NAME, split)
    if version_width_str:
      version_width_regex = '.{%s}' % version_width_str

    split_glob_pattern = re.sub(VERSION_FULL_REGEX, '*', split_glob_pattern)
    version_capture_regex = '(?P<{}>{})'.format(VERSION_PROPERTY_NAME,
                                                version_width_regex)
    split_regex_pattern = re.sub(VERSION_FULL_REGEX, version_capture_regex,
                                 split_regex_pattern)

  return split_glob_pattern, split_regex_pattern


def _get_target_span_version(
    uri: str,
    split: example_gen_pb2.Input.Split,
    range_config: Optional[range_config_pb2.RangeConfig] = None
) -> Tuple[Optional[int], Optional[int]]:
  """Retrieves a  target span and version for a given split pattern.

  If both Span and Version spec occur in the split pattern, searches for and
  returns both the target Span and Version. If only Span exists in the split
  pattern, searches for the target Span, and Version is returned as None.
  If Version is present, but not Span, an error is raised. If neither Span
  nor Version is present, returns both as None.

  Additonally, supports parsing span number from date stamps using the Date.
  specs. Once the calendar date is parsed from the Date specs, it is converted
  into a span number by counting the number of days since 01/01/1970.

  Args:
    uri: The base path from which files will be searched.
    split: An example_gen_pb2.Input.Split object which contains a split pattern,
      to be searched on.
    range_config: An instance of range_config_pb2.RangeConfig, which specifies
      which spans to consider when finding the most recent span and version. If
      unset, search for latest span number with no restrictions.

  Returns:
    Tuple of two ints, Span (optional) and Version (optional). Note
      that this function will update the {SPAN} or Date tags as well as the
      {VERSION} tags in the split config to actual Span and Version numbers.

  Raises:
    ValueError: if any of the following occurs:
      - If either Span or Version spec is occurs in the split pattern
        more than once.
      - If Version spec is provided, but Span spec is not present.
      - If Span or Version found is not an integer.
      - If a matching cannot be found for split pattern provided.
  """
  is_match_span, is_match_date, is_match_version = verify_split_pattern_specs(
      split)

  if not is_match_span and not is_match_date:
    return (None, None)

  split_glob_pattern, split_regex_pattern = _create_matching_glob_and_regex(
      uri=uri,
      split=split,
      is_match_span=is_match_span,
      is_match_date=is_match_date,
      is_match_version=is_match_version,
      range_config=range_config)

  logging.info('Glob pattern for split %s: %s', split.name, split_glob_pattern)
  logging.info('Regex pattern for split %s: %s', split.name,
               split_regex_pattern)

  latest_span_tokens = None
  latest_span_int = None
  latest_version = None
  latest_version_int = None

  files = fileio.glob(split_glob_pattern)
  for file_path in files:
    match_span_tokens, match_span_int, match_version, match_version_int = (
        _find_matched_span_version_from_path(file_path, split_regex_pattern,
                                             is_match_span, is_match_date,
                                             is_match_version))

    if latest_span_int is None or match_span_int > latest_span_int:
      # Uses str instead of int because of zero padding digits.
      latest_span_tokens = match_span_tokens
      latest_span_int = match_span_int
      latest_version = match_version
      latest_version_int = match_version_int
    elif (latest_span_int == match_span_int and
          (latest_version is None or match_version_int >= latest_version_int)):
      latest_version = match_version
      latest_version_int = match_version_int

  if latest_span_int is None or (is_match_version and latest_version is None):
    raise ValueError('Cannot find matching for split %s based on %s' %
                     (split.name, split.pattern))

  # Update split pattern so executor can find the files to ingest.
  if is_match_span:
    split.pattern = re.sub(SPAN_FULL_REGEX, latest_span_tokens[0],
                           split.pattern)
  elif is_match_date:
    for spec, value in zip(DATE_SPECS, latest_span_tokens):
      split.pattern = split.pattern.replace(spec, value)

  if is_match_version:
    split.pattern = re.sub(VERSION_FULL_REGEX, latest_version, split.pattern)

  return latest_span_int, latest_version_int


def calculate_splits_fingerprint_span_and_version(
    input_base_uri: str,
    splits: Iterable[example_gen_pb2.Input.Split],
    range_config: Optional[range_config_pb2.RangeConfig] = None
) -> Tuple[str, int, Optional[int]]:
  """Calculates the fingerprint of files in a URI matching split patterns.

  If a pattern has the {SPAN} placeholder or the Date spec placeholders, {YYYY},
  {MM}, and {DD}, and optionally, the {VERSION} placeholder, attempts to find
  aligned values that results in all splits having the target span and most
  recent version for that span.

  Args:
    input_base_uri: The base path from which files will be searched.
    splits: An iterable collection of example_gen_pb2.Input.Split objects.
    range_config: An instance of range_config_pb2.RangeConfig, which specifies
      which spans to consider when finding the most recent span and version. If
      unset, search for latest span number with no restrictions.

  Returns:
    A Tuple of [fingerprint, select_span, select_version], where select_span
    is either the value matched with the {SPAN} placeholder, the value mapped
    from matching the calendar date with the date placeholders {YYYY}, {MM},
    {DD} or 0 if a placeholder wasn't specified, and where select_version is
    either the value matched with the {VERSION} placeholder, or None if the
    placeholder wasn't specified. Note that this function will update the
    {SPAN} or Date tags as well as the {VERSION} tags in the split configs to
    actual Span and Version numbers.
  """

  split_fingerprints = []
  select_span = 0
  select_version = None
  # Calculate the fingerprint of files under input_base_uri.
  for split in splits:
    logging.info('select span and version = (%s, %s)', select_span,
                 select_version)
    # Find most recent span and version for this split.
    target_span, target_version = _get_target_span_version(
        input_base_uri, split, range_config=range_config)

    # TODO(b/162622803): add default behavior for when version spec not present.
    target_span = target_span or 0

    logging.info('latest span and version = (%s, %s)', target_span,
                 target_version)

    if select_span == 0 and select_version is None:
      select_span = target_span
      select_version = target_version

    # Check if latest span and version are the same over all splits.
    if select_span != target_span:
      raise ValueError('Latest span should be the same for each split')
    if select_version != target_version:
      raise ValueError('Latest version should be the same for each split')

    # Calculate fingerprint.
    pattern = os.path.join(input_base_uri, split.pattern)
    split_fingerprint = io_utils.generate_fingerprint(split.name, pattern)
    split_fingerprints.append(split_fingerprint)

  fingerprint = '\n'.join(split_fingerprints)
  return fingerprint, select_span, select_version
