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
"""Generic TFX ExampleGen custom driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Any, Dict, List, Text

import absl
import tensorflow as tf

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.types import channel_utils
from tfx.utils import io_utils

# Fingerprint custom property.
_FINGERPRINT = 'input_fingerprint'
# Span custom property.
_SPAN = 'span'
# Span spec used in split pattern.
_SPAN_SPEC = '{SPAN}'


class Driver(base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, it registers external file path as
  an artifact, e.g., for CsvExampleGen and ImportExampleGen.
  """

  def _glob_to_regex(self, glob_pattern: Text) -> Text:
    """Changes glob pattern to regex pattern."""
    regex_pattern = glob_pattern
    regex_pattern = regex_pattern.replace('.', '\\.')
    regex_pattern = regex_pattern.replace('+', '\\+')
    regex_pattern = regex_pattern.replace('*', '[^/]*')
    regex_pattern = regex_pattern.replace('?', '[^/]')
    regex_pattern = regex_pattern.replace('(', '\\(')
    regex_pattern = regex_pattern.replace(')', '\\)')
    return regex_pattern

  def _retrieve_latest_span(self, uri: Text,
                            split: example_gen_pb2.Input.Split) -> Text:
    split_pattern = os.path.join(uri, split.pattern)
    assert split_pattern.count(
        _SPAN_SPEC) == 1, 'Only one {SPAN} is allowed in %s' % (
            split_pattern)

    split_glob_pattern = split_pattern.replace(_SPAN_SPEC, '*')
    absl.logging.info('Glob pattern for split %s: %s' %
                      (split.name, split_glob_pattern))
    split_regex_pattern = self._glob_to_regex(split_pattern).replace(
        _SPAN_SPEC, '(.*)')
    absl.logging.info('Regex pattern for split %s: %s' %
                      (split.name, split_regex_pattern))
    assert re.compile(
        split_regex_pattern).groups == 1, 'Regex should have only one group'

    files = tf.io.gfile.glob(split_glob_pattern)
    latest_span = None
    for file_path in files:
      result = re.search(split_regex_pattern, file_path)
      assert result is not None, ('Glob pattern does not match regex pattern')
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

  def resolve_input_artifacts(
      self,
      input_channels: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Overrides BaseDriver.resolve_input_artifacts()."""
    del driver_args  # unused
    del pipeline_info  # unused

    input_config = example_gen_pb2.Input()
    json_format.Parse(exec_properties['input_config'], input_config)

    input_dict = channel_utils.unwrap_channel_dict(input_channels)
    for input_list in input_dict.values():
      for single_input in input_list:
        absl.logging.debug('Processing input %s.' % single_input.uri)
        absl.logging.debug('single_input %s.' % single_input)
        absl.logging.debug('single_input.mlmd_artifact %s.' %
                           single_input.mlmd_artifact)

        # Set the fingerprint of input.
        split_fingerprints = []
        select_span = None
        for split in input_config.splits:
          # If SPAN is specified, pipeline will process the latest span, note
          # that this span number must be the same for all splits and it will
          # be stored in metadata as the span of input artifact.
          if _SPAN_SPEC in split.pattern:
            latest_span = self._retrieve_latest_span(single_input.uri, split)
            if select_span is None:
              select_span = latest_span
            if select_span != latest_span:
              raise ValueError(
                  'Latest span should be the same for each split: %s != %s' %
                  (select_span, latest_span))
            split.pattern = split.pattern.replace(_SPAN_SPEC, select_span)

          pattern = os.path.join(single_input.uri, split.pattern)
          split_fingerprints.append(
              io_utils.generate_fingerprint(split.name, pattern))
        fingerprint = '\n'.join(split_fingerprints)
        single_input.set_string_custom_property(_FINGERPRINT, fingerprint)
        if select_span is None:
          select_span = '0'
        single_input.set_string_custom_property(_SPAN, select_span)

        matched_artifacts = []
        for artifact in self._metadata_handler.get_artifacts_by_uri(
            single_input.uri):
          if (artifact.custom_properties[_FINGERPRINT].string_value ==
              fingerprint) and (artifact.custom_properties[_SPAN].string_value
                                == select_span):
            matched_artifacts.append(artifact)

        if matched_artifacts:
          # TODO(b/138845899): consider use span instead of id.
          # If there are multiple matches, get the latest one for caching.
          # Using id because spans are the same for matched artifacts.
          latest_artifact = max(
              matched_artifacts, key=lambda artifact: artifact.id)
          absl.logging.debug('latest_artifact %s.' % (latest_artifact))
          absl.logging.debug('type(latest_artifact) %s.' %
                             type(latest_artifact))

          single_input.set_mlmd_artifact(latest_artifact)
        else:
          # TODO(jyzhao): whether driver should be read-only for metadata.
          self._metadata_handler.publish_artifacts([single_input])
          absl.logging.debug('Registered new input: %s' % single_input)

    exec_properties['input_config'] = json_format.MessageToJson(
        input_config, sort_keys=True, preserving_proto_field_name=True)
    return input_dict
