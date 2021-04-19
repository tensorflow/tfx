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
"""Tests for tfx.components.infra_validator.executor."""

import os
import signal
import threading
from typing import Any, Dict, Text
from unittest import mock

import tensorflow as tf
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import executor
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import serving_bins
from tfx.dsl.io import fileio
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import BLESSING_KEY
from tfx.types.standard_component_specs import EXAMPLES_KEY
from tfx.types.standard_component_specs import MODEL_KEY
from tfx.types.standard_component_specs import REQUEST_SPEC_KEY
from tfx.types.standard_component_specs import SERVING_SPEC_KEY
from tfx.types.standard_component_specs import VALIDATION_SPEC_KEY
from tfx.utils import path_utils
from tfx.utils import proto_utils

from google.protobuf import json_format


def _make_serving_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.ServingSpec:
  result = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(payload, result)
  return result


def _make_validation_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.ValidationSpec:
  result = infra_validator_pb2.ValidationSpec()
  json_format.ParseDict(payload, result)
  return result


def _make_request_spec(
    payload: Dict[Text, Any]) -> infra_validator_pb2.RequestSpec:
  result = infra_validator_pb2.RequestSpec()
  json_format.ParseDict(payload, result)
  return result


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    # Setup Mocks

    patcher = mock.patch.object(request_builder, 'build_requests')
    self.build_requests_mock = patcher.start()
    self.addCleanup(patcher.stop)

    # Setup directories

    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    base_output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                     self.get_temp_dir())
    output_data_dir = os.path.join(base_output_dir, self._testMethodName)

    # Setup input_dict.

    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(source_data_dir, 'trainer', 'current')
    self._model_path = path_utils.serving_model_path(self._model.uri)
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'transform',
                                'transformed_examples', 'eval')
    examples.split_names = artifact_utils.encode_split_names(['eval'])

    self._input_dict = {
        MODEL_KEY: [self._model],
        EXAMPLES_KEY: [examples],
    }
    self._blessing = standard_artifacts.InfraBlessing()
    self._blessing.uri = os.path.join(output_data_dir, 'blessing')
    self._output_dict = {BLESSING_KEY: [self._blessing]}
    temp_dir = os.path.join(output_data_dir, '.temp')
    self._context = executor.Executor.Context(tmp_dir=temp_dir, unique_id='1')
    self._serving_spec = _make_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.15.0']
        },
        'local_docker': {},
        'model_name': 'chicago-taxi',
    })
    self._serving_binary = serving_bins.parse_serving_binaries(
        self._serving_spec)[0]
    self._validation_spec = _make_validation_spec({
        'max_loading_time_seconds': 10,
        'num_tries': 3
    })
    self._request_spec = _make_request_spec({
        'tensorflow_serving': {
            'signature_names': ['serving_default'],
        },
        'num_examples': 1
    })
    self._exec_properties = {
        SERVING_SPEC_KEY: proto_utils.proto_to_json(self._serving_spec),
        VALIDATION_SPEC_KEY: proto_utils.proto_to_json(self._validation_spec),
        REQUEST_SPEC_KEY: proto_utils.proto_to_json(self._request_spec),
    }

  def testDo_BlessedIfNoError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce'):
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check blessed.
    self.assertBlessed()

  def testDo_NotBlessedIfError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # Validation will raise error.
      validate_mock.side_effect = ValueError
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check not blessed.
    self.assertNotBlessed()

  def testDo_BlessedIfEventuallyNoError(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # Validation will raise error at first, succeeded at the following.
      # Infra validation will be tried 3 times, so 2 failures are tolerable.
      validate_mock.side_effect = [ValueError, ValueError, None]
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check blessed.
    self.assertBlessed()

  def testDo_NotBlessedIfErrorContinues(self):
    # Run executor.
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(infra_validator, '_ValidateOnce') as validate_mock:
      # 3 Errors are not tolerable.
      validate_mock.side_effect = [ValueError, ValueError, ValueError, None]
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    # Check not blessed.
    self.assertNotBlessed()

  def testDo_MakeSavedModelWarmup(self):
    infra_validator = executor.Executor(self._context)
    self._request_spec.make_warmup = True
    self._exec_properties[REQUEST_SPEC_KEY] = (
        proto_utils.proto_to_json(self._request_spec))

    with mock.patch.object(infra_validator, '_ValidateOnce'):
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    warmup_file = path_utils.warmup_file_path(
        path_utils.stamped_model_path(self._blessing.uri))
    self.assertFileExists(warmup_file)
    self.assertEqual(self._blessing.get_int_custom_property('has_model'), 1)

  def testDo_WarmupNotCreatedWithoutRequests(self):
    infra_validator = executor.Executor(self._context)
    del self._exec_properties[REQUEST_SPEC_KEY]  # No request

    with mock.patch.object(infra_validator, '_ValidateOnce'):
      infra_validator.Do(self._input_dict, self._output_dict,
                         self._exec_properties)

    warmup_file = path_utils.warmup_file_path(
        path_utils.stamped_model_path(self._blessing.uri))
    self.assertFileDoesNotExist(warmup_file)

  def testValidateOnce_LoadOnly_Succeed(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary, 'MakeClient'):
      with mock.patch.object(executor, '_create_model_server_runner'):
        # Should not raise any error.
        infra_validator._ValidateOnce(
            model_path=self._model_path,
            serving_binary=self._serving_binary,
            serving_spec=self._serving_spec,
            validation_spec=self._validation_spec,
            requests=[])

  def testValidateOnce_LoadOnly_FailIfRunnerWaitRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary, 'MakeClient'):
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_runner = mock_runner_factory.return_value
        mock_runner.WaitUntilRunning.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model_path=self._model_path,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=[])

  def testValidateOnce_LoadOnly_FailIfClientWaitRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_client.WaitUntilModelLoaded.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model_path=self._model_path,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=[])
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()

  def testValidateOnce_LoadAndQuery_Succeed(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        infra_validator._ValidateOnce(
            model_path=self._model_path,
            serving_binary=self._serving_binary,
            serving_spec=self._serving_spec,
            validation_spec=self._validation_spec,
            requests=['my_request'])
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()
        mock_client.WaitUntilModelLoaded.assert_called()
        mock_client.SendRequests.assert_called()

  def testValidateOnce_LoadAndQuery_FailIfSendRequestsRaises(self):
    infra_validator = executor.Executor(self._context)
    with mock.patch.object(self._serving_binary,
                           'MakeClient') as mock_client_factory:
      mock_client = mock_client_factory.return_value
      with mock.patch.object(
          executor, '_create_model_server_runner') as mock_runner_factory:
        mock_client.SendRequests.side_effect = ValueError
        with self.assertRaises(ValueError):
          infra_validator._ValidateOnce(
              model_path=self._model_path,
              serving_binary=self._serving_binary,
              serving_spec=self._serving_spec,
              validation_spec=self._validation_spec,
              requests=['my_request'])
        mock_runner_factory.return_value.WaitUntilRunning.assert_called()
        mock_client.WaitUntilModelLoaded.assert_called()

  def testSignalHandling(self):
    infra_validator = executor.Executor(self._context)
    ready_to_kill_event = threading.Event()

    def send_sigterm(pid):
      ready_to_kill_event.wait()
      os.kill(pid, signal.SIGTERM)

    def validate_side_effect(*args, **kwargs):
      del args, kwargs  # Unused.
      ready_to_kill_event.set()
      while True:  # Wait until killed.
        pass

    with mock.patch.object(infra_validator, '_ValidateOnce') as mock_validate:
      mock_validate.side_effect = validate_side_effect
      killer = threading.Thread(target=send_sigterm, args=(os.getpid(),))
      killer.start()

      # SIGTERM should raise GracefulShutdown error.
      with self.assertRaises(error_types.GracefulShutdown):
        infra_validator.Do(
            self._input_dict,
            self._output_dict,
            self._exec_properties)

  def assertBlessed(self):
    self.assertFileExists(os.path.join(self._blessing.uri, 'INFRA_BLESSED'))
    self.assertEqual(1, self._blessing.get_int_custom_property('blessed'))

  def assertNotBlessed(self):
    self.assertFileExists(os.path.join(self._blessing.uri, 'INFRA_NOT_BLESSED'))
    self.assertEqual(0, self._blessing.get_int_custom_property('blessed'))

  def assertFileExists(self, path: Text):
    self.assertTrue(fileio.exists(path))

  def assertFileDoesNotExist(self, path: Text):
    self.assertFalse(fileio.exists(path))

if __name__ == '__main__':
  tf.test.main()
