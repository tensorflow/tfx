# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.tools.cli.handler.kubeflow_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import tarfile
import tempfile
import mock
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import kubeflow_handler
from tfx.utils import io_utils


def _MockSubprocess(cmd, env):  # pylint: disable=invalid-name, unused-argument
  # Store pipeline_args in a pickle file
  pipeline_args_path = env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH]
  pipeline_args = {'pipeline_name': 'chicago_taxi_pipeline_kubeflow'}
  with open(pipeline_args_path, 'w') as f:
    json.dump(pipeline_args, f)

  chicago_taxi_pipeline_dir = os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
  pipeline_path = os.path.join(chicago_taxi_pipeline_dir,
                               'test_pipeline_kubeflow_1.py')
  # Store pipeline package
  output_filename = os.path.join(chicago_taxi_pipeline_dir,
                                 'chicago_taxi_pipeline_kubeflow.tar.gz')
  with tarfile.open(output_filename, 'w:gz') as tar:
    tar.add(pipeline_path)


class _MockResponse(object):
  """Mock upload response object."""

  def __init__(self, config):
    self.host = config['host']
    self.client_id = config['client_id']
    self.namespace = config['namespace']
    self.id = config['id']
    self.name = config['name']


class _MockClientClass(object):

  def __init__(self, host, client_id, namespace):

    self.config = {
        'host': host,
        'client_id': client_id,
        'namespace': namespace,
        'id': 'fake_pipeline_id',
        'name': 'fake_pipeline_name'
    }  # pylint: disable=invalid-name, unused-variable
    self._output_dir = os.path.join(tempfile.gettempdir(), 'output_dir')
    self._pipelines_api = _MockPipelineApi()

  def upload_pipeline(self, pipeline_package_path, pipeline_name):  # pylint: disable=invalid-name, unused-argument
    io_utils.copy_file(
        pipeline_package_path,
        os.path.join(self._output_dir, os.path.basename(pipeline_package_path)),
        overwrite=True)
    return _MockResponse(self.config)

  def list_pipelines(self):
    pass


class _MockPipelineApi(object):

  def delete_pipeline(self, id):  # pylint: disable=redefined-builtin, invalid-name
    pass

  def get_pipeline(self, id):  # pylint: disable=redefined-builtin, invalid-name
    return id


class KubeflowHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(KubeflowHandlerTest, self).setUp()
    self._home = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = self._home
    self._original_kubeflow_home_value = os.environ.get('KUBEFLOW_HOME', '')
    os.environ['KUBEFLOW_HOME'] = os.path.join(os.environ['HOME'], 'kubeflow')

    # Flags for handler.
    self.engine = 'kubeflow'
    self.chicago_taxi_pipeline_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testdata')
    self.pipeline_path = os.path.join(self.chicago_taxi_pipeline_dir,
                                      'test_pipeline_kubeflow_1.py')
    self.pipeline_name = 'chicago_taxi_pipeline_kubeflow'
    self.pipeline_package_path = os.path.join(
        self.chicago_taxi_pipeline_dir, 'chicago_taxi_pipeline_kubeflow.tar.gz')

    # Kubeflow client params.
    self.endpoint = 'dummyEndpoint'
    self.namespace = 'kubeflow'
    self.iap_client_id = 'dummyID'

    # Pipeline args for mocking subprocess.
    self.pipeline_args = {'pipeline_name': 'chicago_taxi_pipeline_kubeflow'}

  def tearDown(self):
    super(KubeflowHandlerTest, self).tearDown()
    os.environ['HOME'] = self._original_home_value
    os.environ['KUBEFLOW_HOME'] = self._original_kubeflow_home_value

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_save_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler._save_pipeline(self.pipeline_args)
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertTrue(os.path.join(handler_pipeline_path, 'pipeline_args.json'))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_create_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))
    handler.create_pipeline()
    self.assertTrue(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_create_pipeline_existent_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()
    # Run create_pipeline again to test.
    with self.assertRaises(SystemExit) as err:
      handler.create_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline {} already exists.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_update_pipeline(self):
    # First create pipeline with test_pipeline.py
    pipeline_path_1 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_1.py')
    flags_dict_1 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_1,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict_1)
    handler.create_pipeline()

    # Update test_pipeline and run update_pipeline
    pipeline_path_2 = os.path.join(self.chicago_taxi_pipeline_dir,
                                   'test_pipeline_kubeflow_2.py')
    flags_dict_2 = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: pipeline_path_2,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict_2)
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertTrue(tf.io.gfile.exists(handler_pipeline_path))
    handler.update_pipeline()
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(handler_pipeline_path, 'pipeline_args.json')))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_update_pipeline_no_pipeline(self):
    # Update pipeline without creating one.
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.update_pipeline()
    self.assertEqual(
        str(err.exception), 'Pipeline {} does not exist.'.format(
            self.pipeline_args[labels.PIPELINE_NAME]))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_compile_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.captureWritesToStream(sys.stdout) as captured:
      handler.compile_pipeline()
    self.assertIn('Pipeline compiled successfully', captured.contents())
    self.assertIn('Pipeline package path', captured.contents())

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_delete_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_DSL_PATH: self.pipeline_path,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
        labels.PIPELINE_PACKAGE_PATH: self.pipeline_package_path
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.create_pipeline()

    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    handler.delete_pipeline()
    handler_pipeline_path = handler._get_handler_pipeline_path(
        self.pipeline_args[labels.PIPELINE_NAME])
    self.assertFalse(tf.io.gfile.exists(handler_pipeline_path))

  @mock.patch('kfp.Client', _MockClientClass)
  @mock.patch('subprocess.call', _MockSubprocess)
  def test_delete_pipeline_non_existent_pipeline(self):
    flags_dict = {
        labels.ENGINE_FLAG: self.engine,
        labels.PIPELINE_NAME: self.pipeline_name,
        labels.ENDPOINT: self.endpoint,
        labels.IAP_CLIENT_ID: self.iap_client_id,
        labels.NAMESPACE: self.namespace,
    }
    handler = kubeflow_handler.KubeflowHandler(flags_dict)
    with self.assertRaises(SystemExit) as err:
      handler.delete_pipeline()
    self.assertEqual(
        str(err.exception),
        'Pipeline {} does not exist.'.format(flags_dict[labels.PIPELINE_NAME]))


if __name__ == '__main__':
  tf.test.main()
