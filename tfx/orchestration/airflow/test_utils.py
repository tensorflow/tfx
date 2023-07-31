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
"""Common utility for testing airflow-based orchestrator."""

import subprocess
import time
from typing import Tuple

from absl import logging
import docker

_MYSQL_POLLING_INTERVAL_SEC = 2
_MYSQL_POLLING_MAX_ATTEMPTS = 60
_MYSQL_PORT = '3306/tcp'


def create_mysql_container(container_name: str) -> Tuple[str, int]:
  """Create a mysql docker container and returns a tuple of ip address and port to it.

  A created mysql will have 'airflow' database and 'tfx' user without password.

  Args:
      container_name: A name of the new container.

  Returns:
      The ip address and new port number.

  Raises:
      RuntimeError: When mysql couldn't respond in pre-defined time limit or
                    failed to run initialization sqls.
  """

  client = docker.from_env()
  container = client.containers.run(
      'mysql:5.7',
      name=container_name,
      environment=['MYSQL_ROOT_PASSWORD=root'],
      ports={_MYSQL_PORT: None},
      detach=True)
  container.reload()  # required to get auto-assigned ports
  port = int(container.ports[_MYSQL_PORT][0]['HostPort'])

  # From go/kokoro-native-docker-migration, the Ubuntu 20.04 version has an
  # issue with connecting to the localhost. The container IP address is manually
  # fetched with the suggested command on the doc.
  command = '/sbin/ip route|awk \'/default/ { print $3 }\''
  ip_address = subprocess.check_output(
      command, shell=True).decode('utf-8').strip()
  logging.info('Container IP address: %s', ip_address)

  for _ in range(_MYSQL_POLLING_MAX_ATTEMPTS):
    logging.info('Waiting for mysqld container...')
    time.sleep(_MYSQL_POLLING_INTERVAL_SEC)
    # MySQL availability should be checked with a network access to distinguish
    # a temporary server with a real mysql. See
    # https://github.com/docker-library/mysql/blob/bc6e37a2bed792b1c4fc6ab1ec3ce316e6a5f061/5.7/docker-entrypoint.sh#L360-L362
    check_available = subprocess.run(  # pylint: disable=subprocess-run-check
        [
            'mysql',
            '-uroot',
            '-proot',
            '-h',
            ip_address,
            '-P',
            str(port),
            '-e',
            'SELECT 1;',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if check_available.returncode == 0:
      break
  else:
    logging.error('Logs from mysql container:\n%s', container.logs())
    raise RuntimeError(
        'MySql could not started in %d seconds'
        % (_MYSQL_POLLING_INTERVAL_SEC * _MYSQL_POLLING_MAX_ATTEMPTS)
    )

  create_db_sql = """
      CREATE USER 'tfx'@'%' IDENTIFIED BY '';
      GRANT ALL ON *.* TO 'tfx'@'%' WITH GRANT OPTION;
      FLUSH PRIVILEGES;
      SET GLOBAL explicit_defaults_for_timestamp = 1;
      CREATE DATABASE airflow;
  """
  exit_code, output_bytes = container.exec_run('mysql -uroot -proot -e "%s"' %
                                               create_db_sql)
  if exit_code != 0:
    output = output_bytes.decode('utf-8')
    logging.error('Failed to run sql for initialization:\n%s', output)
    raise RuntimeError('Failed to run initialization SQLs: {}'.format(output))

  client.close()
  return ip_address, port


def delete_mysql_container(container_name: str):
  """Delete a mysql docker container with name.

  Args:
      container_name: A name of the new container.
  """
  client = docker.from_env()
  container = client.containers.get(container_name)
  container.remove(force=True)
  client.close()


class AirflowScheduler:
  """Launch Airflow scheduler in the context."""

  def __init__(self):
    self._args = ['airflow', 'scheduler']
    self._sub_process = None

  def __enter__(self):
    self._sub_process = subprocess.Popen(self._args)
    while True:
      time.sleep(10)
      check_result = subprocess.run(  # pylint: disable=subprocess-run-check
          ['airflow', 'jobs', 'check', '--job-type', 'SchedulerJob'])
      if check_result.returncode == 0:
        time.sleep(10)  # Wait for a bit more until dag processing is completed.
        break
    return self

  def __exit__(self, exception_type, exception_value, traceback):  # pylint: disable=unused-argument
    if self._sub_process:
      self._sub_process.terminate()
