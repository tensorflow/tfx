# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Text, Tuple, Union

from absl import logging
import six
import tensorflow as tf

from tfx.proto import example_gen_pb2
from tfx.utils import io_utils
from google.protobuf import json_format

# Key for `input_base` in executor exec_properties.
INPUT_BASE_KEY = 'input_base'
# Key for `input_config` in executor exec_properties.
INPUT_CONFIG_KEY = 'input_config'
# Key for `output_config` in executor exec_properties.
OUTPUT_CONFIG_KEY = 'output_config'
# Key for the `output_data_format` in executor exec_properties.
OUTPUT_DATA_FORMAT_KEY = 'output_data_format'

# Key for output examples in executor output_dict.
EXAMPLES_KEY = 'examples'

# Key for the `payload_format` custom property of output examples artifact.
PAYLOAD_FORMAT_PROPERTY_NAME = 'payload_format'
# Key for the `input_fingerprint` custom property of output examples artifact.
FINGERPRINT_PROPERTY_NAME = 'input_fingerprint'
# Key for the `span` custom property of output examples artifact.
SPAN_PROPERTY_NAME = 'span'
# Span spec used in split pattern.
SPAN_SPEC = '{SPAN}'
# Key for the `version` custom property of output examples artifact.
VERSION_PROPERTY_NAME = 'version'
# Version spec used in split pattern.
VERSION_SPEC = '{VERSION}'

_DEFAULT_ENCODING = 'utf-8'


def dict_to_example(instance: Dict[Text, Any]) -> tf.train.Example:
  """Converts dict to tf example."""
  feature = {}
  for key, value in instance.items():
    # TODO(jyzhao): support more types.
    if value is None:
      feature[key] = tf.train.Feature()
    elif isinstance(value, six.integer_types):
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, float):
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[value]))
    elif isinstance(value, six.text_type) or isinstance(value, str):
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(
              value=[value.encode(_DEFAULT_ENCODING)]))
    elif isinstance(value, list):
      if not value:
        feature[key] = tf.train.Feature()
      elif isinstance(value[0], six.integer_types):
        feature[key] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=value))
      elif isinstance(value[0], float):
        feature[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=value))
      elif isinstance(value[0], six.text_type) or isinstance(value[0], str):
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
    input_config: Union[example_gen_pb2.Input, Dict[Text, Any]],
    output_config: Union[example_gen_pb2.Output, Dict[Text,
                                                      Any]]) -> List[Text]:
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
    split_pattern: Text = '*') -> example_gen_pb2.Input:
  """Returns default input config."""
  # Treats input base dir as a single split.
  return example_gen_pb2.Input(splits=[
      example_gen_pb2.Input.Split(name='single_split', pattern=split_pattern)
  ])


def make_default_output_config(
    input_config: Union[example_gen_pb2.Input, Dict[Text, Any]]
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


def _glob_to_regex(glob_pattern: Text) -> Text:
  """Changes glob pattern to regex pattern."""
  regex_pattern = glob_pattern
  regex_pattern = regex_pattern.replace('.', '\\.')
  regex_pattern = regex_pattern.replace('+', '\\+')
  regex_pattern = regex_pattern.replace('*', '[^/]*')
  regex_pattern = regex_pattern.replace('?', '[^/]')
  regex_pattern = regex_pattern.replace('(', '\\(')
  regex_pattern = regex_pattern.replace(')', '\\)')
  return regex_pattern


def _retrieve_latest_span_version(
    uri: Text, split: example_gen_pb2.Input.Split
) -> Tuple[Optional[Text], Optional[Text]]:
  """Retrieves the most recent span and version for a given split pattern.

  If both Span and Version spec occur in the split pattern, searches for and
  returns both the latest Span and Version. If only Span exists in the split
  pattern, searches for the latest Span, and Version is returned as None.
  If Version is present, but not Span, an error is raised. If neither Span
  nor Version is present, returns both as None.

  Args:
    uri: The base path from which files will be searched.
    split: An example_gen_pb2.Input.Split object which contains a split pattern,
      to be searched on.

  Returns:
    Tuple of two strings, Span (optional) and Version (optional).

  Raises:
    ValueError: if any of the following occurs:
      - If either Span or Version spec is occurs in the split pattern
        more than once.
      - If Version spec is provided, but Span spec is not present.
      - If Span or Version found is not an integer.
      - If a matching cannot be found for split pattern provided.
  """

  split_pattern = os.path.join(uri, split.pattern)

  split_glob_pattern = split_pattern
  split_regex_pattern = _glob_to_regex(split_pattern)

  latest_span = None
  latest_version = None

  if SPAN_SPEC not in split.pattern:
    if VERSION_SPEC in split.pattern:
      raise ValueError('Version spec provided, but Span spec is not present')
    return latest_span, latest_version

  if split.pattern.count(SPAN_SPEC) != 1:
    raise ValueError('Only one %s is allowed in %s' %
                     (SPAN_SPEC, split.pattern))

  split_glob_pattern = split_glob_pattern.replace(SPAN_SPEC, '*')
  split_regex_pattern = split_regex_pattern.replace(
      SPAN_SPEC, '(?P<{}>.*)'.format(SPAN_PROPERTY_NAME))

  is_match_version = VERSION_SPEC in split.pattern
  if is_match_version:
    if split.pattern.count(VERSION_SPEC) != 1:
      raise ValueError('Only one %s is allowed in %s' %
                       (VERSION_SPEC, split.pattern))
    split_glob_pattern = split_glob_pattern.replace(VERSION_SPEC, '*')
    split_regex_pattern = split_regex_pattern.replace(
        VERSION_SPEC, '(?P<{}>.*)'.format(VERSION_PROPERTY_NAME))

  logging.info('Glob pattern for split %s: %s', split.name, split_glob_pattern)
  logging.info('Regex pattern for split %s: %s', split.name,
               split_regex_pattern)

  files = tf.io.gfile.glob(split_glob_pattern)

  for file_path in files:
    result = re.search(split_regex_pattern, file_path)
    if result is None:
      raise ValueError('Glob pattern does not match regex pattern')

    span_str = result.group(SPAN_PROPERTY_NAME)
    try:
      span_int = int(span_str)
    except ValueError:
      raise ValueError('Cannot find %s number from %s based on %s' %
                       (SPAN_PROPERTY_NAME, file_path, split_regex_pattern))

    version_str = None
    if is_match_version:
      version_str = result.group(VERSION_PROPERTY_NAME)
      try:
        version_int = int(version_str)
      except ValueError:
        raise ValueError(
            'Cannot find %s number from %s based on %s' %
            (VERSION_PROPERTY_NAME, file_path, split_regex_pattern))

    if latest_span is None or span_int > int(latest_span):
      # Uses str instead of int because of zero padding digits.
      latest_span = span_str
      latest_version = version_str
    elif (span_int == int(latest_span) and
          (latest_version is None or version_int >= int(latest_version))):
      latest_version = version_str

  if latest_span is None or (is_match_version and latest_version is None):
    raise ValueError('Cannot find matching for split %s based on %s' %
                     (split.name, split.pattern))

  return latest_span, latest_version


def calculate_splits_fingerprint_span_and_version(
    input_base_uri: Text, splits: Iterable[example_gen_pb2.Input.Split]
) -> Tuple[Text, Text, Optional[Text]]:
  """Calculates the fingerprint of files in a URI matching split patterns.

  If a pattern has the {SPAN} placeholder and, optionally, the {VERSION}
  placeholder, attempts to find aligned values that results in all splits
  having the most recent span and most recent version for that span.

  Args:
    input_base_uri: The base path from which files will be searched.
    splits: An iterable collection of example_gen_pb2.Input.Split objects. Note
      that this function will update the {SPAN} in this and {VERSION} tags in
      the split config to actual Span and Version numbers.

  Returns:
    A Tuple of [fingerprint, select_span, select_version], where select_span
    is either the value matched with the {SPAN} placeholder, or '0' if the
    placeholder wasn't specified, and where select_version is either the
    value matched with the {VERSION} placeholder, or None if the placeholder
    wasn't specified.
  """

  split_fingerprints = []
  select_span = '0'
  select_version = None
  # Calculate the fingerprint of files under input_base_uri.
  for split in splits:
    logging.info('select span and version = (%s, %s)', select_span,
                 select_version)
    # Find most recent span and version for this split.
    latest_span, latest_version = _retrieve_latest_span_version(
        input_base_uri, split)

    # Replace split.pattern so executor can find files after driver runs.
    if latest_span:
      split.pattern = split.pattern.replace(SPAN_SPEC, latest_span)
    if latest_version:
      split.pattern = split.pattern.replace(VERSION_SPEC, latest_version)

    # TODO(b/162622803): add default behavior for when version spec not present.
    latest_span = latest_span or '0'

    logging.info('latest span and version = (%s, %s)', latest_span,
                 latest_version)

    if select_span == '0' and select_version is None:
      select_span = latest_span
      select_version = latest_version

    # Check if latest span and version are the same over all splits.
    if select_span != latest_span:
      raise ValueError('Latest span should be the same for each split')
    if select_version != latest_version:
      raise ValueError('Latest version should be the same for each split')

    # Calculate fingerprint.
    pattern = os.path.join(input_base_uri, split.pattern)
    split_fingerprint = io_utils.generate_fingerprint(split.name, pattern)
    split_fingerprints.append(split_fingerprint)

  fingerprint = '\n'.join(split_fingerprints)
  return fingerprint, select_span, select_version
