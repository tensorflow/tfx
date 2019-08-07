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
import tensorflow as tf
from typing import Any, Dict, List, Text
from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.types import channel_utils
from tfx.utils import io_utils

# Fingerprint custom property.
FINGERPRINT = 'input_fingerprint'


class Driver(base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, it registers external file path as
  an artifact, e.g., for CsvExampleGen and ImportExampleGen.
  """

  # TODO(ruoyu): Deprecate this in favor of resolve_input_artifacts once
  # migration to go/tfx-oss-artifacts-passing finishes.
  def _prepare_input_for_processing(
      self,
      input_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any],
  ) -> Dict[Text, List[types.Artifact]]:
    """Resolves artifacts for external inputs."""
    input_config = example_gen_pb2.Input()
    json_format.Parse(exec_properties['input_config'], input_config)

    for input_list in input_dict.values():
      for single_input in input_list:
        tf.logging.info('Processing input {}.'.format(single_input.uri))
        tf.logging.info('single_input {}.'.format(single_input))
        tf.logging.info('single_input.artifact {}.'.format(
            single_input.artifact))

        # Set the fingerprint of input.
        split_fingerprints = []
        for split in input_config.splits:
          pattern = os.path.join(single_input.uri, split.pattern)
          split_fingerprints.append(
              io_utils.generate_fingerprint(split.name, pattern))
        fingerprint = '\n'.join(split_fingerprints)
        single_input.set_string_custom_property(FINGERPRINT, fingerprint)

        matched_artifacts = []
        for artifact in self._metadata_handler.get_artifacts_by_uri(
            single_input.uri):
          if (artifact.custom_properties[FINGERPRINT].string_value ==
              fingerprint):
            matched_artifacts.append(artifact)

        if matched_artifacts:
          # TODO(b/138845899): consider use span instead of id.
          # If there are multiple matches, get the latest one for caching.
          # Using id because spans are the same for matched artifacts.
          latest_artifact = max(
              matched_artifacts, key=lambda artifact: artifact.id)
          tf.logging.info('latest_artifact {}.'.format(latest_artifact))
          tf.logging.info('type(latest_artifact) {}.'.format(
              type(latest_artifact)))

          single_input.set_artifact(latest_artifact)
        else:
          # TODO(jyzhao): whether driver should be read-only for metadata.
          [new_artifact] = self._metadata_handler.publish_artifacts(
              [single_input])  # pylint: disable=unbalanced-tuple-unpacking
          tf.logging.info('Registered new input: {}'.format(new_artifact))
          single_input.set_artifact(new_artifact)

    return input_dict

  def resolve_input_artifacts(
      self,
      input_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
  ) -> Dict[Text, List[types.Artifact]]:
    """Overrides BaseDriver.resolve_input_artifacts()."""
    del driver_args  # unused
    return self._prepare_input_for_processing(
        channel_utils.unwrap_channel_dict(input_dict), exec_properties)
