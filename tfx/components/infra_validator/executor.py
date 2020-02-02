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
import grpc
from typing import cast, Any, Dict, List, Optional, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator.model_server_runners import factory
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import path_utils

# Filename of infra blessing artifact on succeed.
BLESSED = 'INFRA_BLESSED'
# Filename of infra blessing artifact on fail.
NOT_BLESSED = 'INFRA_NOT_BLESSED'


def _is_query_mode(input_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> bool:
  return 'examples' in input_dict and 'request_spec' in exec_properties


class Executor(base_executor.BaseExecutor):
  """TFX infra validator executor."""

  def __init__(self,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    super(Executor, self).__init__(context)
    self._validation_failed = False

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
          result. It is an empty file with the name either of INFRA_BLESSED or
          INFRA_NOT_BLESSED.
      exec_properties:
        - `serving_spec`: Serialized `ServingSpec` configuration.
        - `validation_spec`: Serialized `ValidationSpec` configuration.
        - `request_spec`: Serialized `RequestSpec` configuration.
    """
    model = artifact_utils.get_single_instance(input_dict['model'])
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])

    serving_spec = infra_validator_pb2.ServingSpec()
    json_format.Parse(exec_properties['serving_spec'], serving_spec)
    validation_spec = infra_validator_pb2.ValidationSpec()
    json_format.Parse(exec_properties['validation_spec'], validation_spec)

    if _is_query_mode(input_dict, exec_properties):
      logging.info('InfraValidator will be run in LOAD_AND_QUERY mode.')
      request_spec = infra_validator_pb2.RequestSpec()
      json_format.Parse(exec_properties['request_spec'], request_spec)
      examples = artifact_utils.get_single_instance(input_dict['examples'])
      requests = request_builder.build_requests(
          model_name=os.path.basename(
              os.path.dirname(path_utils.serving_model_path(model.uri))),
          examples=examples,
          request_spec=request_spec)
    else:
      logging.info('InfraValidator will be run in LOAD_ONLY mode.')
      requests = []

    runners = factory.create_model_server_runners(
        model=cast(standard_artifacts.Model, model),
        serving_spec=serving_spec)

    # TODO(jjong): Make logic parallel.
    for runner in runners:
      with _defer_stop(runner):
        logging.info('Starting %s.', repr(runner))
        runner.Start()

        # Check model is successfully loaded.
        if not runner.WaitUntilModelAvailable(
            timeout_secs=validation_spec.max_loading_time_seconds):
          logging.error('Failed to load model in %s; marking as not blessed.',
                        repr(runner))
          self._MarkNotBlessed(blessing)
          continue

        # Check model can be successfully queried.
        if requests:
          try:
            runner.client.IssueRequests(requests)
          except (grpc.RpcError, ValueError) as e:
            logging.error(e)
            logging.error(
                'Failed to query model in %s; marking as not blessed.',
                repr(runner))
            self._MarkNotBlessed(blessing)
            continue

    self._MarkBlessedIfSucceeded(blessing)

  def _MarkBlessedIfSucceeded(self, blessing: types.Artifact) -> None:
    if not self._validation_failed:
      logging.info('Model passed infra validation; marking model as blessed.')
      io_utils.write_string_file(os.path.join(blessing.uri, BLESSED), '')
      blessing.set_int_custom_property('blessed', 1)

  def _MarkNotBlessed(self, blessing: types.Artifact) -> None:
    if not self._validation_failed:
      self._validation_failed = True
      io_utils.write_string_file(os.path.join(blessing.uri, NOT_BLESSED), '')
      blessing.set_int_custom_property('blessed', 0)


@contextlib.contextmanager
def _defer_stop(stoppable):
  try:
    yield
  finally:
    logging.info('Stopping %s.', repr(stoppable))
    stoppable.Stop()
