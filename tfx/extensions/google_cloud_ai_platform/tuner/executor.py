# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Helper class to start TFX Tuner as a Job on Google Cloud AI Platform."""

import datetime
import json
import multiprocessing
import os
from typing import Any, Dict, List

from absl import logging
from tfx import types
from tfx.components.tuner import executor as tuner_executor
from tfx.dsl.components.base import base_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.types import standard_component_specs
from tfx.utils import doc_controls
from tfx.utils import json_utils

TUNING_ARGS_KEY = doc_controls.documented(
    obj='ai_platform_tuning_args',
    doc='Keys to the items in custom_config of Tuner for passing tuning args '
    'to AI Platform.')

REMOTE_TRIALS_WORKING_DIR_KEY = doc_controls.documented(
    obj='remote_trials_working_dir',
    doc='Keys to the items in custom_config of Tuner for specifying a working '
    'dir for remote trial.')

# Directory to store intermediate hyperparamter search progress.
# TODO(b/160188053): Use the same temp dir as the calling Executor.
_WORKING_DIRECTORY = '/tmp'


class Executor(base_executor.BaseExecutor):
  """Tuner executor that launches parallel tuning flock on Cloud AI Platform.

  This executor starts a Cloud AI Platform (CAIP) Training job with a flock of
  workers, where each worker independently executes Tuner's search loop on
  the single machine.

  Per KerasTuner's design, distributed Tuner's identity is controlled by the
  environment variable (KERASTUNER_TUNER_ID) to each workers in the CAIP
  training job. Those environment variables are configured in each worker of
  CAIP training job's worker flock.

  In addition, some implementation of KerasTuner requires a separate process
  to centrally manage the state of tuning (called as 'chief oracle') which is
  consulted by all workers according as another set of environment variables
  (KERASTUNER_ORACLE_IP and KERASTUNER_ORACLE_PORT).

  In summary, distributed tuning flock by Cloud AI Platform Job is structured
  as follows.

  Executor.Do() -> launch _Executor.Do() on a possibly multi-worker CAIP job ->

    -+> master -> _search() (-> create a subprocess -> run the chief oracle.)
     |                       +> trigger a single tuner.search()
     +> worker -> _search()  -> trigger a single tuner.search()
     +> worker -> _search()  -> trigger a single tuner.search()
  """

  # TODO(b/160013376): Refactor common parts with Trainer Executor.
  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Starts a Tuner component as a job on Google Cloud AI Platform."""
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(standard_component_specs.CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is None:
      raise ValueError('custom_config is not provided')

    if not isinstance(custom_config, Dict):
      raise TypeError('custom_config in execution properties must be a dict, '
                      'but received %s' % type(custom_config))

    training_inputs = custom_config.get(TUNING_ARGS_KEY)
    if training_inputs is None:
      err_msg = ('\'%s\' not found in custom_config.' % TUNING_ARGS_KEY)
      logging.error(err_msg)
      raise ValueError(err_msg)
    training_inputs = training_inputs.copy()

    tune_args = tuner_executor.get_tune_args(exec_properties)

    num_parallel_trials = (1
                           if not tune_args else tune_args.num_parallel_trials)
    if num_parallel_trials > 1:
      # Chief node is also responsible for conducting tuning loop.
      desired_worker_count = num_parallel_trials - 1

      if training_inputs.get('workerCount') != desired_worker_count:
        logging.warning('workerCount is overridden with %s',
                        desired_worker_count)
        training_inputs['workerCount'] = desired_worker_count

      training_inputs['scaleTier'] = 'CUSTOM'
      training_inputs['masterType'] = (
          training_inputs.get('masterType') or 'standard')
      training_inputs['workerType'] = (
          training_inputs.get('workerType') or 'standard')

    # 'tfx_tuner_YYYYmmddHHMMSS' is the default job ID if not specified.
    job_id = (
        custom_config.get(ai_platform_trainer_executor.JOB_ID_KEY) or
        'tfx_tuner_{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))

    # TODO(b/160059039): Factor out label creation to a utility function.
    executor_class = _WorkerExecutor
    executor_class_path = '%s.%s' % (executor_class.__module__,
                                     executor_class.__name__)

    # Note: exec_properties['custom_config'] here is a dict.
    return runner.start_cloud_training(input_dict, output_dict, exec_properties,
                                       executor_class_path, training_inputs,
                                       job_id)


def _need_chief_oracle(exec_properties: Dict[str, Any]) -> bool:
  """Returns True if the Tuner instance requires a chief oracle."""
  # TODO(b/160902662): Skip chief oracle for CloudTuner that does not require
  #                    chief oracle for distributed tuning (it is a no-op,
  #                    because it simply forwards to the AI Platform Optimizer
  #                    service).
  del exec_properties
  return True


class _WorkerExecutor(base_executor.BaseExecutor):
  """TFX Tuner executor impl as a worker in a Google Cloud AI Platform job."""

  def _start_chief_oracle_in_subprocess(
      self, input_dict: Dict[str, List[types.Artifact]],
      exec_properties: Dict[str, List[types.Artifact]]):
    """Starts a chief oracle in a subprocess."""

    def _run_chief_oracle() -> None:
      """Invoke chief oracle, and listen to the open port."""
      logging.info('chief_oracle() starting...')

      # Per KerasTuner's specification, configuration of chief oracle is set
      # by environment variables. This only affects the current sub-process
      # which is single-threaded, but not the main process. As such, mutation
      # of this otherwise global state is safe.
      os.environ['KERASTUNER_ORACLE_IP'] = '0.0.0.0'
      os.environ['KERASTUNER_ORACLE_PORT'] = self._master_port
      os.environ['KERASTUNER_TUNER_ID'] = 'chief'

      logging.info('Binding chief oracle server at: %s:%s',
                   os.environ['KERASTUNER_ORACLE_IP'],
                   os.environ['KERASTUNER_ORACLE_PORT'])

      # By design of KerasTuner, chief oracle blocks forever. Ref.
      # https://github.com/keras-team/keras-tuner/blob/e8b0ad3ecae471c73e17cb41f37e6f99202ac0dd/kerastuner/engine/base_tuner.py#L74-L76
      tuner_executor.search(input_dict, exec_properties, _WORKING_DIRECTORY)

    # Because of KerasTuner's interface whereby behavior is controlled
    # by environment variables, starting the chief oracle in a sub-process,
    # as opposed to another thread in the main process, in order not to leak
    # the environment variables.
    result = multiprocessing.Process(target=_run_chief_oracle)
    result.start()

    logging.info('Chief oracle started at PID: %s', result.pid)
    return result

  def _search(self, input_dict: Dict[str, List[types.Artifact]],
              exec_properties: Dict[str, List[types.Artifact]]):
    """Conducts a single search loop, setting up chief oracle if necessary."""

    # If not distributed, simply conduct search and return.
    if self._tuner_id is None:
      return tuner_executor.search(input_dict, exec_properties,
                                   _WORKING_DIRECTORY)

    if _need_chief_oracle(exec_properties):

      # If distributed search, and this node is chief, start a chief oracle
      # process before conducting search by itself.
      if self._is_chief:
        # Tuner with chief oracle will block forever. As such, start it in
        # a subprocess and manage its lifecycle by the main process.
        # Note that the Tuner with chief oracle does not run search loop,
        # hence does not run TensorFlow code in the subprocess.
        self._chief_process = self._start_chief_oracle_in_subprocess(
            input_dict, exec_properties)

      # If distributed, both master and worker need to know where the oracle is.
      # Per KerasTuner's interface, it is configured through env variables.
      # This only affects the current main process, which is designed to be
      # single-threaded. As such, mutation of this otherwise global state is
      # safe.
      os.environ['KERASTUNER_ORACLE_IP'] = self._master_addr
      os.environ['KERASTUNER_ORACLE_PORT'] = self._master_port

      logging.info('Oracle chief is known to be at: %s:%s',
                   os.environ['KERASTUNER_ORACLE_IP'],
                   os.environ['KERASTUNER_ORACLE_PORT'])

    # Conduct tuner search loop, regardless of master or worker.
    # There is only one Tuner instance in the current process, as such,
    # controllling the id of the Tuner instance via environment variable
    # is safe.
    os.environ['KERASTUNER_TUNER_ID'] = self._tuner_id
    logging.info('Setting KERASTUNER_TUNER_ID with %s',
                 os.environ['KERASTUNER_TUNER_ID'])

    return tuner_executor.search(input_dict, exec_properties,
                                 _WORKING_DIRECTORY)

  def __init__(self, context):
    super().__init__(context)

    # Those fields are populated only when running in distribution.
    self._is_chief = False
    self._tuner_id = None
    self._master_addr = None
    self._master_port = None

    self._chief_process = None  # Populated when the chief oracle is started.

    # Initialize configuration of distribution according to CLUSTER_SPEC
    logging.info('Initializing cluster spec... ')

    cluster_spec = json.loads(os.environ.get('CLUSTER_SPEC', '{}'))

    # If CLUSTER_SPEC is not present, assume single-machine tuning.
    if not cluster_spec:
      return

    self._master_addr, self._master_port = (
        # We rely on Cloud AI Platform Training service's specification whereby
        # there will be no more than one master replica.
        # https://cloud.google.com/ai-platform/training/docs/distributed-training-containers#cluster-spec-format
        cluster_spec['cluster']['master'][0].split(':'))

    self._tuner_id = (
        'tfx-tuner-%s-%d' % (
            cluster_spec['task']['type'],  # 'master' or 'worker'
            cluster_spec['task']['index']  # zero-based index
        ))

    logging.info('Tuner ID is: %s', self._tuner_id)

    self._is_chief = cluster_spec['task']['type'] == 'master'

    logging.info('Cluster spec initalized with: %s', cluster_spec)

  def __del__(self):
    self._close()

  def _close(self) -> None:
    """Kills the chief oracle sub-process, if still running."""
    if self._chief_process and self._chief_process.is_alive():
      logging.info('Terminating chief oracle at PID: %s',
                   self._chief_process.pid)
      self._chief_process.terminate()

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:

    tuner = self._search(input_dict, exec_properties)

    if self._tuner_id is not None and not self._is_chief:
      logging.info('Returning since this is not chief worker.')
      return

    tuner_executor.write_best_hyperparameters(tuner, output_dict)

    self._close()
