# Lint as: python2, python3
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
"""Tests for tfx.tools.cli.cmd.helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile

import mock
import tensorflow.compat.v1 as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import airflow_handler
from tfx.tools.cli.handler import beam_handler
from tfx.tools.cli.handler import handler_factory


class _MockClientClass(object):

  def __init__(self, host, client_id, namespace):
    config = {'host': host, 'client_id': client_id, 'namespace': namespace}  # pylint: disable=invalid-name, unused-variable
    self._output_dir = os.path.join(tempfile.gettempdir(), 'output_dir')


class HandlerFactoryTest(tf.test.TestCase):

  def setUp(self):
    super(HandlerFactoryTest, self).setUp()
    self.flags_dict = {}
    sys.modules['kfp'] = mock.Mock()
    sys.modules['kfp_server_api'] = mock.Mock()

  def _MockSubprocessAirflow(self):
    return b'absl-py==0.7.1\nalembic==0.9.10\napache-beam==2.12.0\napache-airflow==1.10.3\n'

  @mock.patch('subprocess.check_output', _MockSubprocessAirflow)
  def testCreateHandlerAirflow(self):
    self.flags_dict[labels.ENGINE_FLAG] = 'airflow'
    self.assertIsInstance(
        handler_factory.create_handler(self.flags_dict),
        airflow_handler.AirflowHandler)

  def _MockSubprocessKubeflow(self):
    return b'absl-py==0.7.1\nadal==1.2.1\nalembic==0.9.10\napache-beam==2.12.0\nkfp==0.1\n'

  @mock.patch('subprocess.check_output', _MockSubprocessKubeflow)
  @mock.patch('kfp.Client', _MockClientClass)
  def testCreateHandlerKubeflow(self):
    flags_dict = {
        labels.ENGINE_FLAG: 'kubeflow',
        labels.ENDPOINT: 'dummyEndpoint',
        labels.IAP_CLIENT_ID: 'dummyID',
        labels.NAMESPACE: 'kubeflow',
    }
    from tfx.tools.cli.handler import kubeflow_handler  # pylint: disable=g-import-not-at-top
    self.assertIsInstance(
        handler_factory.create_handler(flags_dict),
        kubeflow_handler.KubeflowHandler)

  def testCreateHandlerBeam(self):
    self.flags_dict[labels.ENGINE_FLAG] = 'beam'
    self.assertIsInstance(
        handler_factory.create_handler(self.flags_dict),
        beam_handler.BeamHandler)

  def testCreateHandlerOther(self):
    self.flags_dict[labels.ENGINE_FLAG] = 'flink'
    with self.assertRaises(Exception) as err:
      handler_factory.create_handler(self.flags_dict)
    self.assertEqual(
        str(err.exception), 'Engine {} is not supported.'.format(
            self.flags_dict[labels.ENGINE_FLAG]))

  def _MockSubprocessNoEngine(self):
    return b'absl-py==0.7.1\nalembic==0.9.10\napache-beam==2.12.0\n'

  @mock.patch('subprocess.check_output', _MockSubprocessNoEngine)
  def testDetectHandlerMissing(self):
    self.flags_dict[labels.ENGINE_FLAG] = 'auto'
    self.assertIsInstance(
        handler_factory.detect_handler(self.flags_dict),
        beam_handler.BeamHandler)

  def _MockSubprocessMultipleEngines(self):
    return b'absl-py==0.7.1\nadal==1.2.1\nalembic==0.9.10\napache-airflow==1.10.3\napache-beam==2.12.0\nkfp==0.1\n'

  @mock.patch('subprocess.check_output', _MockSubprocessMultipleEngines)
  def testDetectHandlerMultiple(self):
    self.flags_dict[labels.ENGINE_FLAG] = 'auto'
    with self.assertRaises(SystemExit) as cm:
      handler_factory.detect_handler(self.flags_dict)
    self.assertEqual(
        str(cm.exception),
        'Multiple orchestrators found. Choose one using --engine flag.'
        )

if __name__ == '__main__':
  tf.test.main()
