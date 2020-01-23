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
"""TFX InfraValidator executor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os

from absl import logging
from typing import cast, Any, Dict, List, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.infra_validator.model_server_runners import factory
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils

Model = standard_artifacts.Model


class Executor(base_executor.BaseExecutor):
  """TFX infra validator executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Contract for running InfraValidator Executor.

    Args:
      input_dict:
        - `model`: Single `Model` artifact that we're validating.
        - `examples`: `Examples` artifacts to be used for test requests.
      output_dict:
        - `blessing`: Single `InfraBlessing` artifact containing the validated
          result. It is an empty file with the name either of BLESSED or
          NOT_BLESSED.
      exec_properties:
        - `serving_spec`: Serialized `ServingSpec` configuration.
        - `validation_spec`: Serialized `ValidationSpec` configuration.
    """
    model = artifact_utils.get_single_instance(input_dict['model'])
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])
    serving_spec = json_format.Parse(exec_properties['serving_spec'],
                                     infra_validator_pb2.ServingSpec())
    validation_spec = json_format.Parse(exec_properties['validation_spec'],
                                        infra_validator_pb2.ValidationSpec())

    runners = factory.create_model_server_runners(
        cast(Model, model), serving_spec)
    requests = self._BuildRequests(input_dict)

    # TODO(jjong): Make logic parallel.
    for runner in runners:
      with _defer_stop(runner):
        logging.info('Starting %s.', repr(runner))
        runner.Start()

        # Check model is successfully loaded.
        if not runner.WaitUntilModelAvailable(
            timeout_secs=validation_spec.max_loading_time_seconds):
          logging.info('Failed to load model in %s; marking as not blessed.',
                       repr(runner))
          self._Unbless(blessing)
          return

        # Check model can be successfully queried.
        if requests is not None:
          if not runner.client.IssueRequests(requests):
            logging.info('Failed to query model in %s; marking as not blessed.',
                         repr(runner))
            self._Unbless(blessing)
            return

        # Check validation spec.
        if not self._CheckValidationSpec():
          logging.info('Model failed to pass threshold; marking as not blessed')
          self._Unbless(blessing)
          return

    logging.info('Model passed infra validation; marking model as blessed.')
    self._Bless(blessing)

  def _CheckValidationSpec(self):
    """TODO(jjong): use metric collect context to check validation spec."""
    return True

  def _BuildRequests(self, input_dict):
    """TODO(jjong): Fetch requests proto from Requests artifact."""
    if 'examples' in input_dict:
      raise NotImplementedError('Build Request proto from Examples artifact.')
    # Return None if no buildable requests source found.
    return

  def _Bless(self, blessing):
    io_utils.write_string_file(os.path.join(blessing.uri, 'BLESSED'), '')
    blessing.set_int_custom_property('blessed', 1)

  def _Unbless(self, blessing):
    io_utils.write_string_file(os.path.join(blessing.uri, 'NOT_BLESSED'), '')
    blessing.set_int_custom_property('blessed', 0)


@contextlib.contextmanager
def _defer_stop(stoppable):
  try:
    yield
  finally:
    logging.info('Stopping %s.', repr(stoppable))
    stoppable.Stop()
