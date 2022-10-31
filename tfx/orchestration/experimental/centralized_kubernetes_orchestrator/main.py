# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Centralized Kubernetes Orchestrator `main`."""

from concurrent import futures
import contextlib
import time

from absl import app
from absl import flags
from absl import logging
import grpc
from tfx.orchestration import metadata
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator import kubernetes_task_scheduler
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service import kubernetes_orchestrator_service
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task_manager as tm
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts

FLAGS = flags.FLAGS
_MAX_ACTIVE_TASK_SCHEDULERS_FLAG = flags.DEFINE_integer(
    'tflex_max_active_task_schedulers', 100,
    'Maximum number of active task schedulers.')
_INACTIVITY_TTL_SECS_FLAG = flags.DEFINE_float(
    'tflex_inactivity_ttl_secs', 30, 'Orchestrator inactivity TTL. If set, '
    'orchestrator will exit after ttl seconds of no orchestration activity.')
_DEFAULT_POLLING_INTERVAL_SECS_FLAG = flags.DEFINE_float(
    'tflex_default_polling_interval_secs', 10.0,
    'Default orchestration polling interval.')
_MYSQL_HOST_FLAG = flags.DEFINE_string(
    'mysql_host', '127.0.0.1',
    'The name or network address of the instance of MySQL to connect to.')
_MYSQL_PORT_FLAG = flags.DEFINE_integer(
    'mysql_port', 8888, 'The port MySQL is using to listen for connections.')
_SERVER_PORT_FLAG = flags.DEFINE_integer(
    'server_port', 10000,
    'The port rpc server is using to listen for connections.')
_MYSQL_DATABASE_FLAG = flags.DEFINE_string(
    'mysql_database', '', 'The name of the MySQL database to use.')
_MYSQL_USERNAME_FLAG = flags.DEFINE_string(
    'mysql_username', 'root', 'The MySQL login account being used.')
_MYSQL_PASSWORD_FLAG = flags.DEFINE_string(
    'mysql_password', '', 'The password for the MySQL account being used.')

_TICK_DURATION_SECS = 1.0
_MONITORING_INTERVAL_SECS = 30


def _start_grpc_server(
    servicer: kubernetes_orchestrator_service.KubernetesOrchestratorServicer
) -> grpc.Server:
  """Starts GRPC server."""
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  service_pb2_grpc.add_KubernetesOrchestratorServicer_to_server(
      servicer, server)
  server_creds = grpc.local_server_credentials()
  server.add_secure_port(f'[::]:{_SERVER_PORT_FLAG.value}', server_creds)
  server.start()
  return server


def _create_mlmd_connection():
  """Creates connection for MLMD."""
  connection_config = metadata.mysql_metadata_connection_config(
      host=_MYSQL_HOST_FLAG.value,
      port=_MYSQL_PORT_FLAG.value,
      username=_MYSQL_USERNAME_FLAG.value,
      database=_MYSQL_DATABASE_FLAG.value,
      password=_MYSQL_PASSWORD_FLAG.value)
  return metadata.Metadata(connection_config=connection_config)


def _run() -> None:
  """Runs the main orchestration loop."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(event_observer.init())

    mlmd_handle = stack.enter_context(_create_mlmd_connection())
    orchestrator_servicer = kubernetes_orchestrator_service.KubernetesOrchestratorServicer(
        mlmd_handle)

    server = _start_grpc_server(orchestrator_servicer)
    stack.callback(server.stop, grace=None)

    task_queue = tq.TaskQueue()

    service_job_manager = service_jobs.DummyServiceJobManager()
    task_manager = stack.enter_context(
        tm.TaskManager(
            mlmd_handle,
            task_queue,
            max_active_task_schedulers=_MAX_ACTIVE_TASK_SCHEDULERS_FLAG.value))
    last_active = time.time()

    iteration = 0
    while not _INACTIVITY_TTL_SECS_FLAG.value or time.time(
    ) - last_active <= _INACTIVITY_TTL_SECS_FLAG.value:
      try:
        iteration += 1
        logging.info('Orchestration loop: iteration #%d (since process start).',
                     iteration)
        event_observer.check_active()

        # Last pipeline state change time is useful to decide if wait period
        # between iterations can be short-circuited.
        last_state_change_time_secs = (
            pipeline_state.last_state_change_time_secs())

        if pipeline_ops.orchestrate(mlmd_handle, task_queue,
                                    service_job_manager):
          last_active = time.time()

        time_budget = _DEFAULT_POLLING_INTERVAL_SECS_FLAG.value
        logging.info(
            'Orchestration loop: waiting %s seconds before next iteration.',
            time_budget)
        while time_budget > 0.0:
          # Task manager should never be "done" unless there was an error.
          if task_manager.done():
            if task_manager.exception():
              raise task_manager.exception()
            else:
              raise RuntimeError(
                  'Task manager unexpectedly stalled due to an internal error.')

          # Short-circuit if state change is detected.
          if (pipeline_state.last_state_change_time_secs() >
              last_state_change_time_secs):
            last_state_change_time_secs = (
                pipeline_state.last_state_change_time_secs())
            logging.info(
                'Orchestration loop: detected state change, exiting wait period '
                'early (with %s of %s seconds remaining).', time_budget,
                _DEFAULT_POLLING_INTERVAL_SECS_FLAG.value)
            break

          time_budget = _sleep_tick_duration_secs(time_budget)
      except Exception:  # pylint: disable=broad-except
        logging.exception('Exception in main orchestration loop!')
        raise

    logging.info('Exiting due to no pipeline run in %s seconds',
                 _INACTIVITY_TTL_SECS_FLAG.value)


def _sleep_tick_duration_secs(time_budget: float) -> float:
  """Sleeps and returns new time budget; standalone fn to mock in tests."""
  time.sleep(_TICK_DURATION_SECS)
  return time_budget - _TICK_DURATION_SECS


def _register_task_schedulers() -> None:
  """Registers task schedulers."""
  ts.TaskSchedulerRegistry.register(
      'type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec',
      kubernetes_task_scheduler.KubernetesTaskScheduler)
  ts.TaskSchedulerRegistry.register(
      'type.googleapis.com/tfx.orchestration.executable_spec.BeamExecutableSpec',
      kubernetes_task_scheduler.KubernetesTaskScheduler)


def main(unused_arg):
  logging.set_verbosity(logging.INFO)
  _register_task_schedulers()
  _run()


if __name__ == '__main__':
  app.run(main)
