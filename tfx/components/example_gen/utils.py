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
from typing import Any, Dict, Iterable, List, Text, Tuple, Union

import absl
import six
import tensorflow as tf

from tfx.proto import example_gen_pb2
from tfx.utils import io_utils
from google.protobuf import json_format

# Fingerprint custom property.
FINGERPRINT_PROPERTY_NAME = 'input_fingerprint'
# Span custom property.
SPAN_PROPERTY_NAME = 'span'
# Span spec used in split pattern.
SPAN_SPEC = '{SPAN}'

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


def _retrieve_latest_span(uri: Text,
                          split: example_gen_pb2.Input.Split) -> Text:
  """Retrieves the most recently updated span matching a given split pattern."""
  split_pattern = os.path.join(uri, split.pattern)
  if split_pattern.count(SPAN_SPEC) != 1:
    raise ValueError('Only one {SPAN} is allowed in %s' % split_pattern)

  split_glob_pattern = split_pattern.replace(SPAN_SPEC, '*')
  absl.logging.info('Glob pattern for split %s: %s' %
                    (split.name, split_glob_pattern))
  split_regex_pattern = _glob_to_regex(split_pattern).replace(SPAN_SPEC, '(.*)')
  absl.logging.info('Regex pattern for split %s: %s' %
                    (split.name, split_regex_pattern))
  if re.compile(split_regex_pattern).groups != 1:
    raise ValueError('Regex should have only one group')

  files = tf.io.gfile.glob(split_glob_pattern)
  latest_span = None
  for file_path in files:
    result = re.search(split_regex_pattern, file_path)
    if result is None:
      raise ValueError('Glob pattern does not match regex pattern')
    try:
      span = int(result.group(1))
    except ValueError:
      raise ValueError('Cannot not find span number from %s based on %s' %
                       (file_path, split_regex_pattern))
    if latest_span is None or span >= int(latest_span):
      # Uses str instead of int because of zero padding digits.
      latest_span = result.group(1)

  if latest_span is None:
    raise ValueError('Cannot not find matching for split %s based on %s' %
                     (split.name, split.pattern))
  return latest_span


def calculate_splits_fingerprint_and_span(
    input_base_uri: Text,
    splits: Iterable[example_gen_pb2.Input.Split]) -> Tuple[Text, Any]:
  """Calculates the fingerprint of files in a URI matching split patterns.

  If a pattern has the {SPAN} placeholder, attempts to find an identical value
  across splits that results in all splits having the most recently updated
  files.

  Args:
    input_base_uri: The base path from which files will be searched
    splits: An iterable collection of example_gen_pb2.Input.Split objects

  Returns:
    A Tuple of [fingerprint, select_span], where select_span is either
    the value matched with the {SPAN} placeholder, or None if the placeholder
    wasn't specified.
  """

  split_fingerprints = []
  select_span = None
  # Calculate the fingerprint of files under input_base_uri.
  for split in splits:
    absl.logging.info('select span = %s' % select_span)
    if SPAN_SPEC in split.pattern:
      latest_span = _retrieve_latest_span(input_base_uri, split)
      absl.logging.info('latest span = %s' % latest_span)
      if select_span is None:
        select_span = latest_span
      if select_span != latest_span:
        raise ValueError(
            'Latest span should be the same for each split: %s != %s' %
            (select_span, latest_span))
      split.pattern = split.pattern.replace(SPAN_SPEC, select_span)
    if select_span is None:
      select_span = '0'
    # Calculate fingerprint
    pattern = os.path.join(input_base_uri, split.pattern)
    fingerprint = io_utils.generate_fingerprint(split.name, pattern)
    split_fingerprints.append(fingerprint)
  fingerprint = '\n'.join(split_fingerprints)
  return fingerprint, select_span
