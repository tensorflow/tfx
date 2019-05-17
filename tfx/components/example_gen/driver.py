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

import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx.components.base import base_driver
from tfx.utils import types


class Driver(base_driver.BaseDriver):
  """Custom driver for ExampleGen.

  This driver supports file based ExampleGen, it registers external file path as
  an artifact, e.g., for CsvExampleGen and ImportExampleGen.
  """

  def _prepare_input_for_processing(
      self, input_dict: Dict[Text, List[types.TfxArtifact]]
      ) -> Dict[Text, List[types.TfxArtifact]]:
    """Resolves artifacts for external inputs."""
    # TODO(jyzhao): check state of the artifacts.
    registered_artifacts = self._metadata_handler.get_all_artifacts()

    for input_list in input_dict.values():
      for single_input in input_list:
        tf.logging.info('Processing input {}.'.format(single_input.uri))
        tf.logging.info('single_input {}.'.format(single_input))
        tf.logging.info('single_input.artifact {}.'.format(
            single_input.artifact))
        matched_artifacts = [
            artifact for artifact in registered_artifacts
            if artifact.uri == single_input.uri
        ]
        if matched_artifacts:
          # If there are multiple matches, get the latest one for caching.
          # Using id because spans are the same for matched artifacts.
          latest_artifact = max(
              matched_artifacts, key=lambda artifact: artifact.id)
          tf.logging.info('latest_artifact {}.'.format(latest_artifact))
          tf.logging.info('type(latest_artifact) {}.'.format(
              type(latest_artifact)))

          single_input.set_artifact(latest_artifact)
        else:
          # TODO(jyzhao): support span.
          single_input.span = 1
          # TODO(jyzhao): whether driver should be read-only for metadata.
          [new_artifact] = self._metadata_handler.publish_artifacts(
              [single_input])  # pylint: disable=unbalanced-tuple-unpacking
          tf.logging.info('Registered new input: {}'.format(new_artifact))
          single_input.set_artifact(new_artifact)

    return input_dict

  def prepare_execution(
      self,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      output_dict: Dict[Text, List[types.TfxArtifact]],
      exec_properties: Dict[Text, Any],
      driver_options: base_driver.DriverOptions,
  ) -> base_driver.ExecutionDecision:
    """Extends BaseDriver by resolving external inputs."""
    updated_input_dict = self._prepare_input_for_processing(input_dict)
    return self._default_caching_handling(updated_input_dict, output_dict,
                                          exec_properties, driver_options)
