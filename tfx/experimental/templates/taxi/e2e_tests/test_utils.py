# Lint as: python2, python3
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
"""E2E test utilities for taxi template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import datetime
import locale
import os
import subprocess
import tarfile
import time

from typing import Text, List, Iterable, Tuple
import urllib.request

from absl import logging
from click import testing as click_testing
import tensorflow as tf

from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils
from tfx.tools.cli.cli_main import cli_group
from tfx.utils import io_utils
from tfx.utils import telemetry_utils
import yaml
from google.cloud import storage


class BaseEndToEndTest(tf.test.TestCase):
  """This test covers step 1~6 of the accompanying document[1] for taxi template.

  [1]https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb
  """

  def setUp(self):
    super(BaseEndToEndTest, self).setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    # TODO(b/150100590) Delete this block after Python >=3.7
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    self._temp_dir = self.create_tempdir().full_path

    self._pipeline_name = 'TAXI_TEMPLATE_E2E_TEST'
    self._project_dir = os.path.join(self._temp_dir, 'src')
    self._old_cwd = os.getcwd()
    os.mkdir(self._project_dir)
    os.chdir(self._project_dir)

    # Initialize CLI runner.
    self._cli_runner = click_testing.CliRunner()

  def tearDown(self):
    super(BaseEndToEndTest, self).tearDown()
    os.chdir(self._old_cwd)

  def _cleanup_with_retry(self, method):
    max_num_trial = 3
    for _ in range(max_num_trial):
      try:
        method()
      except Exception as err:  # pylint:disable=broad-except
        logging.info(err)
      else:
        break

  def _run_cli(self, args: List[Text]) -> click_testing.Result:
    logging.info('Running cli: %s', args)
    result = self._cli_runner.invoke(cli_group, args)
    logging.info('%s', result.output)
    if result.exit_code != 0:
      logging.error('Exit code from cli: %d, exception:%s', result.exit_code,
                    result.exception)
      logging.error('Traceback: %s', result.exc_info)

    return result

  def _add_all_components(self) -> Text:
    """Change 'pipeline.py' file to put all components into the pipeline."""
    return self._uncomment(
        os.path.join('pipeline', 'pipeline.py'), ['components.append('])

  def _uncomment(self, filepath: Text, expressions: Iterable[Text]) -> Text:
    """Update given file by uncommenting the `expression`."""
    replacements = [('# ' + s, s) for s in expressions]
    return self._replace_file_content(filepath, replacements)

  def _replace_file_content(self, filepath: Text,
                            replacements: Iterable[Tuple[Text, Text]]) -> Text:
    """Update given file using `replacements`."""
    path = os.path.join(self._project_dir, filepath)
    with open(path) as fp:
      content = fp.read()
    for old, new in replacements:
      content = content.replace(old, new)
    io_utils.write_string_file(path, content)
    return path

  def _uncomment_multiline_variables(self, filepath: Text,
                                     variables: Iterable[Text]) -> Text:
    """Update given file by uncommenting a variable.

    The variable should be defined in following form.
    # ....
    # VARIABLE_NAME = ...
    #   long indented line
    #
    #   long indented line
    # OTHER STUFF

    Above comments will become

    # ....
    VARIABLE_NAME = ...
      long indented line

      long indented line
    # OTHER STUFF

    Arguments:
      filepath: file to modify.
      variables: List of variables.

    Returns:
      Absolute path of the modified file.
    """
    path = os.path.join(self._project_dir, filepath)
    result = []
    commented_variables = ['# ' + variable + ' =' for variable in variables]
    commented_arguments = ['# ' + variable + '=[' for variable in variables]
    in_variable_definition = False
    in_argument_definition = False

    with open(path) as fp:
      for line in fp:
        if in_variable_definition:
          if line.startswith('#  ') or line.startswith('# }'):
            result.append(line[2:])
            continue
          elif line == '#\n':
            result.append(line[1:])
            continue
          else:
            in_variable_definition = False
        if in_argument_definition:
          print(line)
          if line.lstrip().startswith("# "):
            if "]" in line:
              result.append(line.replace("# ", ""))
              in_argument_definition = False
              continue
            else:
              result.append(line.replace("# ", ""))
              continue

        for commented_var in commented_variables:
          if line.startswith(commented_var):
            in_variable_definition = True
            result.append(line[2:])
            break
        for commented_var in commented_arguments:
          if line.lstrip().startswith(commented_var):
            in_argument_definition = True
            result.append(line.replace("# ", ''))
            break
        else:
          # doesn't include a variable definition to uncomment.
          result.append(line)

    io_utils.write_string_file(path, ''.join(result))
    return path

  def _copy_template(self):
    result = self._run_cli([
        'template',
        'copy',
        '--pipeline_name',
        self._pipeline_name,
        '--destination_path',
        self._project_dir,
        '--model',
        'taxi',
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying taxi pipeline template', result.output)

class KubeflowBaseEndToEndTest(BaseEndToEndTest):
  """This test covers step 1~6 of the accompanying document[1] for taxi template.

  [1]https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb
  """

  def _get_kfp_runs(self):
    # CLI uses experiment_name which is the same as pipeline_name.
    experiment_id = self._kfp_client.get_experiment(
        experiment_name=self._pipeline_name).id
    response = self._kfp_client.list_runs(experiment_id=experiment_id)
    return response.runs

  def _delete_runs(self):
    for run in self._get_kfp_runs():
      self._kfp_client._run_api.delete_run(id=run.id)  # pylint: disable=protected-access

  def _delete_pipeline(self):
    self._run_cli([
        'pipeline', 'delete', '--engine', 'kubeflow', '--pipeline_name',
        self._pipeline_name, '--endpoint', self._endpoint
    ])

  def _delete_pipeline_data(self):
    path = 'tfx_pipeline_output/{}'.format(self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)
    path = '{}/{}'.format(self._DATA_DIRECTORY_NAME, self._pipeline_name)
    orchestration_test_utils.delete_gcs_files(self._GCP_PROJECT_ID,
                                              self._BUCKET_NAME, path)

  def _delete_base_container_image(self):
    subprocess.check_output([
        'gcloud', 'container', 'images', 'delete', self._base_container_image
    ])

  def _delete_target_container_image(self):
    subprocess.check_output([
        'gcloud', 'container', 'images', 'delete', self._target_container_image
    ])

  def _prepare_data(self):
    client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = client.bucket(self._BUCKET_NAME)
    blob = bucket.blob('{}/{}/data.csv'.format(self._DATA_DIRECTORY_NAME,
                                               self._pipeline_name))
    blob.upload_from_filename('data/data.csv')

  def _prepare_base_container_image(self):
    orchestration_test_utils.build_and_push_docker_image(
        self._base_container_image, self._REPO_BASE)

  def _get_endpoint(self):  # pylint: disable=inconsistent-return-statements
    output = subprocess.check_output(
        'kubectl describe configmap inverse-proxy-config -n kubeflow'.split())
    for line in output.decode('utf-8').split('\n'):
      if line.endswith('googleusercontent.com'):
        return line

  def _prepare_skaffold(self):
    self._skaffold = os.path.join(self._temp_dir, 'skaffold')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64',
        self._skaffold)
    os.chmod(self._skaffold, 0o775)

  def _create_pipeline(self):
    result = self._run_cli([
        'pipeline',
        'create',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_dag_runner.py',
        '--endpoint',
        self._endpoint,
        '--build-target-image',
        self._target_container_image,
        '--skaffold-cmd',
        self._skaffold,
        '--build-base-image',
        self._base_container_image,
    ])
    self.assertEqual(0, result.exit_code)

  def _update_pipeline(self):
    result = self._run_cli([
        'pipeline',
        'update',
        '--engine',
        'kubeflow',
        '--pipeline_path',
        'kubeflow_dag_runner.py',
        '--endpoint',
        self._endpoint,
        '--skaffold-cmd',
        self._skaffold,
    ])
    self.assertEqual(0, result.exit_code)

  def _run_pipeline(self):
    result = self._run_cli([
        'run',
        'create',
        '--engine',
        'kubeflow',
        '--pipeline_name',
        self._pipeline_name,
        '--endpoint',
        self._endpoint,
    ])
    self.assertEqual(0, result.exit_code)
    self._wait_until_completed(self._parse_run_id(result.output))

  def _parse_run_id(self, output: str):
    run_id_lines = [
        line for line in output.split('\n')
        if '| {} |'.format(self._pipeline_name) in line
    ]
    self.assertLen(run_id_lines, 1)
    return run_id_lines[0].split('|')[2].strip()

  def _wait_until_completed(self, run_id: str):
    # This timeout will never expire. polling_count * interval == 20min.
    timeout = datetime.timedelta(hours=1)
    end_state = kubeflow_test_utils.poll_kfp_with_retry(
        self._endpoint, run_id, self._MAX_POLLING_COUNT, timeout,
        self._POLLING_INTERVAL_IN_SECONDS)
    self.assertEqual(end_state.lower(), kubeflow_test_utils.KFP_SUCCESS_STATUS)

  def _check_telemetry_label(self):
    file_path = os.path.join(self._project_dir,
                             '{}.tar.gz'.format(self._pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))

    with tarfile.TarFile.open(file_path).extractfile(
        'pipeline.yaml') as pipeline_file:
      self.assertIsNotNone(pipeline_file)
      pipeline = yaml.safe_load(pipeline_file)
      metadata = [
          c['metadata'] for c in pipeline['spec']['templates'] if 'dag' not in c
      ]
      for m in metadata:
        self.assertEqual('tfx-template',
                         m['labels'][telemetry_utils.LABEL_KFP_SDK_ENV])

  def _get_grpc_port(self) -> Text:
    """Get the port number used by MLMD gRPC server."""
    get_grpc_port_command = [
        'kubectl', '-n', 'kubeflow', 'get', 'configmap',
        'metadata-grpc-configmap', '-o',
        'jsonpath={.data.METADATA_GRPC_SERVICE_PORT}'
    ]

    grpc_port = subprocess.check_output(get_grpc_port_command)
    return grpc_port.decode('utf-8')

  def _setup_mlmd_port_forward(self) -> subprocess.Popen:
    """Uses port forward to talk to MLMD gRPC server."""
    grpc_port = self._get_grpc_port()

    is_bind = False

    for port in range(self._KFP_E2E_TEST_FORWARDING_PORT_BEGIN,
                      self._KFP_E2E_TEST_FORWARDING_PORT_END):
      grpc_forward_command = [
          'kubectl', 'port-forward', 'deployment/metadata-grpc-deployment',
          '-n', 'kubeflow', ('%s:%s' % (port, grpc_port))
      ]
      # Begin port forwarding.
      proc = subprocess.Popen(grpc_forward_command)
      try:
        # Wait while port forward to pod is being established
        poll_grpc_port_command = ['lsof', '-i', ':%s' % port]
        result = subprocess.run(  # pylint: disable=subprocess-run-check
            poll_grpc_port_command,
            stdout=subprocess.PIPE)
        for _ in range(self._MAX_ATTEMPTS):
          if (result.returncode == 0 and
              'kubectl' in result.stdout.decode('utf-8')):
            is_bind = True
            break
          logging.info(
              'Waiting while gRPC port-forward is being established...')
          time.sleep(5)
          result = subprocess.run(  # pylint: disable=subprocess-run-check
              poll_grpc_port_command,
              stdout=subprocess.PIPE)

      except:  # pylint: disable=bare-except
        # Kill the process in case unexpected error occurred.
        proc.kill()

      if is_bind:
        self._port = port
        break

    if not is_bind:
      raise RuntimeError('Failed to establish gRPC port-forward to cluster in '
                         'the specified range: port %s to %s' %
                         (self._KFP_E2E_TEST_FORWARDING_PORT_BEGIN,
                          self._KFP_E2E_TEST_FORWARDING_PORT_END))

    return proc
