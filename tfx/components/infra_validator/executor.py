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

import os
import time

from absl import logging
from typing import Any, Dict, List, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.infra_validator import error_types

from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import serving_bins
from tfx.components.infra_validator import types as iv_types
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils.model_paths import tf_serving_flavor

_DEFAULT_NUM_TRIES = 5
_DEFAULT_POLLING_INTERVAL_SEC = 1
_DEFAULT_MAX_LOADING_TIME_SEC = 300
_DEFAULT_MODEL_NAME = 'infra-validation-model'

# Filename of infra blessing artifact on succeed.
BLESSED = 'INFRA_BLESSED'
# Filename of infra blessing artifact on fail.
NOT_BLESSED = 'INFRA_NOT_BLESSED'


def _is_query_mode(input_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> bool:
  return 'examples' in input_dict and 'request_spec' in exec_properties


def _create_model_server_runner(
    model_path: Text,
    serving_binary: serving_bins.ServingBinary,
    serving_spec: infra_validator_pb2.ServingSpec):
  """Create a ModelServerRunner from a model, a ServingBinary and a ServingSpec.

  Args:
    model_path: An IV-flavored model path. (See model_path_utils.py)
    serving_binary: One of ServingBinary instances parsed from the
        `serving_spec`.
    serving_spec: A ServingSpec instance of this infra validation.

  Returns:
    A ModelServerRunner.
  """
  platform = serving_spec.WhichOneof('serving_platform')
  if platform == 'local_docker':
    return local_docker_runner.LocalDockerRunner(
        model_path=model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec
    )
  else:
    raise NotImplementedError('Invalid serving_platform {}'.format(platform))


def _mark_blessed(blessing: types.Artifact) -> None:
  logging.info('Model passed infra validation.')
  io_utils.write_string_file(os.path.join(blessing.uri, BLESSED), '')
  blessing.set_int_custom_property('blessed', 1)


def _mark_not_blessed(blessing: types.Artifact) -> None:
  logging.info('Model failed infra validation.')
  io_utils.write_string_file(os.path.join(blessing.uri, NOT_BLESSED), '')
  blessing.set_int_custom_property('blessed', 0)


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
          result. It is an empty file with the name either of INFRA_BLESSED or
          INFRA_NOT_BLESSED.
      exec_properties:
        - `serving_spec`: Serialized `ServingSpec` configuration.
        - `validation_spec`: Serialized `ValidationSpec` configuration.
        - `request_spec`: Serialized `RequestSpec` configuration.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    model = artifact_utils.get_single_instance(input_dict['model'])
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])

    serving_spec = infra_validator_pb2.ServingSpec()
    json_format.Parse(exec_properties['serving_spec'], serving_spec)
    if not serving_spec.model_name:
      serving_spec.model_name = _DEFAULT_MODEL_NAME

    validation_spec = infra_validator_pb2.ValidationSpec()
    if 'validation_spec' in exec_properties:
      json_format.Parse(exec_properties['validation_spec'], validation_spec)
    if not validation_spec.num_tries:
      validation_spec.num_tries = _DEFAULT_NUM_TRIES
    if not validation_spec.max_loading_time_seconds:
      validation_spec.max_loading_time_seconds = _DEFAULT_MAX_LOADING_TIME_SEC

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

    model_path = self._PrepareModelPath(model.uri, serving_spec)
    try:
      # TODO(jjong): Make logic parallel.
      all_passed = True
      for serving_binary in serving_bins.parse_serving_binaries(serving_spec):
        all_passed &= self._ValidateWithRetry(
            model_path=model_path,
            serving_binary=serving_binary,
            serving_spec=serving_spec,
            validation_spec=validation_spec,
            requests=requests)
    finally:
      io_utils.delete_dir(self._get_tmp_dir())

    if all_passed:
      _mark_blessed(blessing)
    else:
      _mark_not_blessed(blessing)

  def _PrepareModelPath(
      self, model_uri: Text,
      serving_spec: infra_validator_pb2.ServingSpec) -> Text:
    model_path = path_utils.serving_model_path(model_uri)
    serving_binary = serving_spec.WhichOneof('serving_binary')
    if serving_binary == 'tensorflow_serving':
      # TensorFlow Serving requires model to be stored in its own directory
      # structure flavor. If current model_path does not conform to the flavor,
      # we need to make a copy to the temporary path.
      try:
        # Check whether current model_path conforms to the tensorflow serving
        # model path flavor. (Parsed without exception)
        tf_serving_flavor.parse_model_path(
            model_path,
            expected_model_name=serving_spec.model_name)
      except ValueError:
        # Copy the model to comply with the tensorflow serving model path
        # flavor.
        temp_model_path = tf_serving_flavor.make_model_path(
            model_base_path=self._get_tmp_dir(),
            model_name=serving_spec.model_name,
            version=int(time.time()))
        io_utils.copy_dir(src=model_path, dst=temp_model_path)
        return temp_model_path

    return model_path

  def _ValidateWithRetry(
      self, model_path: Text,
      serving_binary: serving_bins.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      requests: List[iv_types.Request]):

    for _ in range(validation_spec.num_tries):
      try:
        self._ValidateOnce(
            model_path=model_path,
            serving_binary=serving_binary,
            serving_spec=serving_spec,
            validation_spec=validation_spec,
            requests=requests)
        # If validation has passed without any exception, succeeded.
        return True
      except Exception as e:  # pylint: disable=broad-except
        # Exception indicates validation failure. Log the error and retry.
        logging.error(e)
        if isinstance(e, error_types.DeadlineExceeded):
          logging.info('Consider increasing the value of '
                       'ValidationSpec.max_loading_time_seconds.')
        continue

    # Every trial has failed. Marking model as not blessed.
    return False

  def _ValidateOnce(
      self, model_path: Text,
      serving_binary: serving_bins.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      requests: List[iv_types.Request]):

    deadline = time.time() + validation_spec.max_loading_time_seconds
    runner = _create_model_server_runner(
        model_path=model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec)

    try:
      logging.info('Starting %r.', runner)
      runner.Start()

      # Check model is successfully loaded.
      runner.WaitUntilRunning(deadline)
      client = serving_binary.MakeClient(runner.GetEndpoint())
      client.WaitUntilModelLoaded(
          deadline, polling_interval_sec=_DEFAULT_POLLING_INTERVAL_SEC)

      # Check model can be successfully queried.
      if requests:
        client.SendRequests(requests)
    finally:
      logging.info('Stopping %r.', runner)
      runner.Stop()
