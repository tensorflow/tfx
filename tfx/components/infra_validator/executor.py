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
from typing import Any, Dict, List, Optional, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import serving_binary_lib
from tfx.components.infra_validator import types as iv_types
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.components.infra_validator.model_server_runners import kubernetes_runner
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import time_utils

_DEFAULT_NUM_TRIES = 2
_DEFAULT_POLLING_INTERVAL_SEC = 1

# Filename of infra blessing artifact on succeed.
BLESSED = 'INFRA_BLESSED'
# Filename of infra blessing artifact on fail.
NOT_BLESSED = 'INFRA_NOT_BLESSED'


def _is_query_mode(input_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> bool:
  return 'examples' in input_dict and 'request_spec' in exec_properties


def _create_model_server_runner(
    standard_model_path: path_utils.StandardModelPath,
    serving_binary: serving_binary_lib.ServingBinary,
    serving_spec: infra_validator_pb2.ServingSpec):
  """Create a ModelServerRunner from a model, a ServingBinary and a ServingSpec.

  Args:
    standard_model_path: A StandardModelPath instance.
    serving_binary: One of ServingBinary instances parsed from the
        `serving_spec`.
    serving_spec: A ServingSpec instance of this infra validation.

  Returns:
    A ModelServerRunner.
  """
  platform = serving_spec.WhichOneof('serving_platform')
  if platform == 'local_docker':
    return local_docker_runner.LocalDockerRunner(
        standard_model_path=standard_model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec
    )
  elif platform == 'kubernetes':
    return kubernetes_runner.KubernetesRunner(
        standard_model_path=standard_model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec
    )
  else:
    raise NotImplementedError('Invalid serving_platform {}'.format(platform))


@contextlib.contextmanager
def _defer_stop(runner: base_runner.BaseModelServerRunner):
  try:
    yield
  finally:
    logging.info('Stopping %s.', repr(runner))
    runner.Stop()


class Executor(base_executor.BaseExecutor):
  """TFX infra validator executor."""

  def __init__(self,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    super(Executor, self).__init__(context)
    self._validation_failed = False
    self._needs_tmp_cleanup = False

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
    validation_spec = infra_validator_pb2.ValidationSpec()
    json_format.Parse(exec_properties['validation_spec'], validation_spec)

    if _is_query_mode(input_dict, exec_properties):
      logging.info('InfraValidator will be run in LOAD_AND_QUERY mode.')
      request_spec = infra_validator_pb2.RequestSpec()
      json_format.Parse(exec_properties['request_spec'], request_spec)
      examples = artifact_utils.get_single_instance(input_dict['examples'])
      requests = request_builder.build_requests(
          model_name=serving_spec.model_name,
          examples=examples,
          request_spec=request_spec)
    else:
      logging.info('InfraValidator will be run in LOAD_ONLY mode.')
      requests = []

    # Get the standard model path. If existing model directory structure is not
    # a standard, copy would be made.
    smp = self._GetStandardModelPath(model.uri, serving_spec.model_name)

    # TODO(jjong): Make logic parallel.
    for serving_binary in serving_binary_lib.parse_serving_binaries(
        serving_spec):
      self._ValidateWithRetry(
          standard_model_path=smp,
          blessing=blessing,
          serving_binary=serving_binary,
          serving_spec=serving_spec,
          validation_spec=validation_spec,
          requests=requests)

    self._MarkBlessedIfSucceeded(blessing)
    self._CleanupTmpIfNecessary()

  def _GetStandardModelPath(self, model_uri: Text,
                            model_name: Text) -> path_utils.StandardModelPath:
    standard_path = path_utils.get_standard_serving_path(model_uri)
    if standard_path:
      return standard_path

    # Make a copy to a standard directory if existing directory structure is not
    # a standard.
    self._needs_tmp_cleanup = True
    model_path = path_utils.serving_model_path(model_uri)
    temp_smp = path_utils.StandardModelPath(
        base_path=self._get_tmp_dir(),
        model_name=model_name,
        version=str(int(time_utils.utc_timestamp())))
    io_utils.copy_dir(src=model_path, dst=temp_smp.full_path)
    return temp_smp

  def _ValidateWithRetry(
      self, standard_model_path: path_utils.StandardModelPath,
      blessing: types.Artifact,
      serving_binary: serving_binary_lib.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      requests: List[iv_types.Request]):

    num_tries = validation_spec.num_tries or _DEFAULT_NUM_TRIES
    for _ in range(num_tries):
      try:
        self._ValidateOnce(
            standard_model_path=standard_model_path,
            serving_binary=serving_binary,
            serving_spec=serving_spec,
            validation_spec=validation_spec,
            requests=requests)
        return
      except Exception as e:  # pylint: disable=broad-except
        logging.error(e)
        continue

    self._MarkNotBlessed(blessing)

  def _ValidateOnce(
      self, standard_model_path: path_utils.StandardModelPath,
      serving_binary: serving_binary_lib.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      requests: List[iv_types.Request]):

    deadline = (time_utils.utc_timestamp()
                + validation_spec.max_loading_time_seconds)
    runner = _create_model_server_runner(
        standard_model_path=standard_model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec)

    with _defer_stop(runner):
      logging.info('Starting %s.', repr(runner))
      runner.Start()

      # Check model is successfully loaded.
      runner.WaitUntilRunning(deadline)
      client = serving_binary.MakeClient(runner.GetEndpoint())
      client.WaitUntilModelLoaded(
          deadline, polling_interval_sec=_DEFAULT_POLLING_INTERVAL_SEC)

      # Check model can be successfully queried.
      if requests:
        client.SendRequests(requests)

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

  def _CleanupTmpIfNecessary(self):
    if self._needs_tmp_cleanup:
      io_utils.delete_dir(self._context.get_tmp_path())
