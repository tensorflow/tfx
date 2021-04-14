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

import contextlib
import functools
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from absl import logging
from tfx import types
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import serving_bins
from tfx.components.infra_validator import types as iv_types
from tfx.components.infra_validator.model_server_runners import kubernetes_runner
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.dsl.components.base import base_executor
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types.standard_component_specs import BLESSING_KEY
from tfx.types.standard_component_specs import EXAMPLES_KEY
from tfx.types.standard_component_specs import MODEL_KEY
from tfx.types.standard_component_specs import REQUEST_SPEC_KEY
from tfx.types.standard_component_specs import SERVING_SPEC_KEY
from tfx.types.standard_component_specs import VALIDATION_SPEC_KEY
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils
from tfx.utils.model_paths import tf_serving_flavor

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2


_DEFAULT_NUM_TRIES = 5
_DEFAULT_POLLING_INTERVAL_SEC = 1
_DEFAULT_MAX_LOADING_TIME_SEC = 300
_DEFAULT_MODEL_NAME = 'infra-validation-model'

# Proto message keys for oneof block.
_TENSORFLOW_SERVING = 'tensorflow_serving'
_LOCAL_DOCKER = 'local_docker'
_KUBERNETES = 'kubernetes'

# Artifact property keys
_BLESSED_KEY = 'blessed'
_MODEL_FLAG_KEY = 'has_model'
# Filename of infra blessing artifact on succeed.
_BLESSED_FILENAME = 'INFRA_BLESSED'
# Filename of infra blessing artifact on fail.
_NOT_BLESSED_FILENAME = 'INFRA_NOT_BLESSED'


def _create_model_server_runner(
    model_path: str,
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
  elif platform == 'kubernetes':
    return kubernetes_runner.KubernetesRunner(
        model_path=model_path,
        serving_binary=serving_binary,
        serving_spec=serving_spec
    )
  else:
    raise NotImplementedError('Invalid serving_platform {}'.format(platform))


def _convert_to_prediction_log(request: iv_types.Request):
  """Try convert infra validation request to TF-Serving PredictionLog."""
  if isinstance(request, classification_pb2.ClassificationRequest):
    return prediction_log_pb2.PredictionLog(
        classify_log=prediction_log_pb2.ClassifyLog(request=request))
  elif isinstance(request, regression_pb2.RegressionRequest):
    return prediction_log_pb2.PredictionLog(
        regress_log=prediction_log_pb2.RegressLog(request=request))
  elif isinstance(request, predict_pb2.PredictRequest):
    return prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
  else:
    raise NotImplementedError(
        f'Cannot convert {type(request)} to PredictionLog')


def _mark_blessed(blessing: types.Artifact) -> None:
  logging.info('Model passed infra validation.')
  io_utils.write_string_file(
      os.path.join(blessing.uri, _BLESSED_FILENAME), '')
  blessing.set_int_custom_property(_BLESSED_KEY, 1)


def _mark_not_blessed(blessing: types.Artifact) -> None:
  logging.info('Model failed infra validation.')
  io_utils.write_string_file(
      os.path.join(blessing.uri, _NOT_BLESSED_FILENAME), '')
  blessing.set_int_custom_property(_BLESSED_KEY, 0)


class Executor(base_executor.BaseExecutor):
  """TFX infra validator executor."""

  def __init__(self,
               context: Optional[base_executor.BaseExecutor.Context] = None):
    super(Executor, self).__init__(context)
    self._cleanups = []

  def _AddCleanup(self, function, *args, **kwargs):
    self._cleanups.append(functools.partial(function, *args, **kwargs))

  def _Cleanup(self):
    for cleanup in self._cleanups:
      try:
        cleanup()
      except:  # pylint: disable=broad-except, bare-except
        logging.warning('Error occurred during cleanup.', exc_info=True)

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Contract for running InfraValidator Executor.

    Args:
      input_dict:
        - `model`: Single `Model` artifact that we're validating.
        - `examples`: `Examples` artifacts to be used for test requests.
      output_dict:
        - `blessing`: Single `InfraBlessing` artifact containing the validated
          result and optinally validated model if warmup requests are appended.
          Artifact URI includes an empty file with the name either of
          INFRA_BLESSED or INFRA_NOT_BLESSED.
      exec_properties:
        - `serving_spec`: Serialized `ServingSpec` configuration.
        - `validation_spec`: Serialized `ValidationSpec` configuration.
        - `request_spec`: Serialized `RequestSpec` configuration.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    model = artifact_utils.get_single_instance(input_dict[MODEL_KEY])
    blessing = artifact_utils.get_single_instance(output_dict[BLESSING_KEY])

    if input_dict.get(EXAMPLES_KEY):
      examples = artifact_utils.get_single_instance(input_dict[EXAMPLES_KEY])
    else:
      examples = None

    serving_spec = infra_validator_pb2.ServingSpec()
    proto_utils.json_to_proto(exec_properties[SERVING_SPEC_KEY], serving_spec)
    if not serving_spec.model_name:
      serving_spec.model_name = _DEFAULT_MODEL_NAME

    validation_spec = infra_validator_pb2.ValidationSpec()
    if exec_properties.get(VALIDATION_SPEC_KEY):
      proto_utils.json_to_proto(exec_properties[VALIDATION_SPEC_KEY],
                                validation_spec)
    if not validation_spec.num_tries:
      validation_spec.num_tries = _DEFAULT_NUM_TRIES
    if not validation_spec.max_loading_time_seconds:
      validation_spec.max_loading_time_seconds = _DEFAULT_MAX_LOADING_TIME_SEC

    if exec_properties.get(REQUEST_SPEC_KEY):
      request_spec = infra_validator_pb2.RequestSpec()
      proto_utils.json_to_proto(exec_properties[REQUEST_SPEC_KEY],
                                request_spec)
    else:
      request_spec = None

    with self._InstallGracefulShutdownHandler():
      self._Do(
          model=model,
          examples=examples,
          blessing=blessing,
          serving_spec=serving_spec,
          validation_spec=validation_spec,
          request_spec=request_spec,
      )

  @contextlib.contextmanager
  def _InstallGracefulShutdownHandler(self):
    # pylint: disable=g-doc-return-or-yield
    """Install graceful shutdown behavior.

    Caveat: InfraValidator currently only recognizes SIGTERM signal as a
    graceful shutdown. Furthermore, SIGTERM can be handled only if the executor
    is running on the MainThread (the thread that runs the python interpreter)
    due to the limitation of Python API.

    When the executor is running on Kubernetes, SIGTERM is a standard way to
    signal the graceful shutdown. Python default behavior for receiving SIGTERM
    is to terminate the process without raising any exception. By registering a
    handler that raises on signal, we can effectively transform the signal to an
    exception, and we can reuse our cleanup code inside "except" or "finally"
    block during the grace period.

    When the executor is run by the local Beam DirectRunner, the executor thread
    is one of the worker threads (not a MainThread) therefore SIGTERM cannot
    be recognized. If either of MainThread or worker thread receives SIGTERM,
    executor will die immediately without grace period.

    Even if the executor fails to shutdown gracefully, external resources that
    are created by model server runner can be cleaned up if the platform
    supports such mechanism (e.g. activeDeadlineSeconds in Kubernetes).
    """

    def _handler(signum, frame):
      del frame  # Unused.
      raise error_types.GracefulShutdown('Got signal {}.'.format(signum))

    try:
      old_handler = signal.signal(signal.SIGTERM, _handler)
    except ValueError:
      # If current thread is not a MainThread, it is not allowed to register
      # the signal handler (ValueError raised).
      logging.info('Unable to register signal handler for non-MainThread '
                   '(name=%s). SIGTERM will not be handled.',
                   threading.current_thread().name)
      old_handler = None

    try:
      yield
    finally:
      self._Cleanup()
      if old_handler:
        signal.signal(signal.SIGTERM, old_handler)

  def _Do(
      self,
      model: types.Artifact,
      examples: Optional[types.Artifact],
      blessing: types.Artifact,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      request_spec: Optional[infra_validator_pb2.RequestSpec],
  ):

    if examples and request_spec:
      logging.info('InfraValidator will be run in LOAD_AND_QUERY mode.')
      requests = request_builder.build_requests(
          model_name=serving_spec.model_name,
          model=model,
          examples=examples,
          request_spec=request_spec)
    else:
      logging.info('InfraValidator will be run in LOAD_ONLY mode.')
      requests = []

    model_path = self._PrepareModelPath(model, serving_spec)
    # TODO(jjong): Make logic parallel.
    all_passed = True
    for serving_binary in serving_bins.parse_serving_binaries(serving_spec):
      all_passed &= self._ValidateWithRetry(
          model_path=model_path,
          serving_binary=serving_binary,
          serving_spec=serving_spec,
          validation_spec=validation_spec,
          requests=requests)

    if all_passed:
      _mark_blessed(blessing)
      if requests and request_spec.make_warmup:
        self._CreateWarmupModel(blessing, model_path, warmup_requests=requests)
    else:
      _mark_not_blessed(blessing)

  def _CreateWarmupModel(self, blessing: types.Artifact, model_path: str,
                         warmup_requests: List[iv_types.Request]):
    output_model_path = path_utils.stamped_model_path(blessing.uri)
    io_utils.copy_dir(src=model_path, dst=output_model_path)
    io_utils.write_tfrecord_file(
        path_utils.warmup_file_path(output_model_path),
        *[_convert_to_prediction_log(r) for r in warmup_requests])
    blessing.set_int_custom_property(_MODEL_FLAG_KEY, 1)

  def _PrepareModelPath(self, model: types.Artifact,
                        serving_spec: infra_validator_pb2.ServingSpec) -> str:
    model_path = path_utils.serving_model_path(
        model.uri, path_utils.is_old_model_artifact(model))
    serving_binary = serving_spec.WhichOneof('serving_binary')
    if serving_binary == _TENSORFLOW_SERVING:
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
        self._AddCleanup(io_utils.delete_dir, self._context.get_tmp_path())
        return temp_model_path

    return model_path

  def _ValidateWithRetry(
      self, model_path: str,
      serving_binary: serving_bins.ServingBinary,
      serving_spec: infra_validator_pb2.ServingSpec,
      validation_spec: infra_validator_pb2.ValidationSpec,
      requests: List[iv_types.Request]):

    for i in range(validation_spec.num_tries):
      logging.info('Starting infra validation (attempt %d/%d).', i + 1,
                   validation_spec.num_tries)
      try:
        self._ValidateOnce(
            model_path=model_path,
            serving_binary=serving_binary,
            serving_spec=serving_spec,
            validation_spec=validation_spec,
            requests=requests)
      except error_types.GracefulShutdown:
        # GracefulShutdown means infra validation aborted. No more retry and
        # escalate the error.
        raise
      except Exception as e:  # pylint: disable=broad-except
        # Other exceptions indicates validation failure. Log the error and
        # retry.
        logging.exception('Infra validation (attempt %d/%d) failed.', i + 1,
                          validation_spec.num_tries)
        if isinstance(e, error_types.DeadlineExceeded):
          logging.info('Consider increasing the value of '
                       'ValidationSpec.max_loading_time_seconds.')
      else:
        # If validation has passed without any exception, succeeded.
        return True

    # Every trial has failed. Marking model as not blessed.
    return False

  def _ValidateOnce(
      self, model_path: str,
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
