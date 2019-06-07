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
"""End to end test for building and running a docker image."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import docker
import pytest
import tensorflow as tf

import unittest


@pytest.mark.end_to_end
class ContainerImageEndToEndTest(unittest.TestCase):
  """An end to end test running TFX container using Kubeflow's entrypoint ."""

  def setUp(self):
    super(ContainerImageEndToEndTest, self).setUp()
    self._image_tag = 'container_image_end_to_end_test_{}:latest'.format(
        self._testMethodName).lower()
    self._client = docker.from_env()
    self._output_basedir = tempfile.mkdtemp()
    repo_base = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    tf.logging.info('Building image %s', self._image_tag)
    _ = self._client.images.build(
        path=repo_base,
        dockerfile='tfx/tools/docker/Dockerfile',
        tag=self._image_tag,
    )

  def tearDown(self):
    super(ContainerImageEndToEndTest, self).tearDown()
    self._client.close()

  def testRunSchemaGen(self):
    command = [
        '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py',
        '--outputs',
        """{"output": [{"artifact": {"uri": "%s", "properties": {"split": {"stringValue": ""}, "type_name": {"stringValue": "SchemaPath"}, "span": {"intValue": "0"}}}, "artifact_type": {"name": "SchemaPath", "properties": {"split": "STRING", "type_name": "STRING", "name": "STRING", "span": "INT", "state": "STRING"}}}]}
        """ % (self._output_basedir),
        '--exec_properties',
        '{"output_dir": "%s"}' % (self._output_basedir),
        '--executor_class_path=tfx.components.schema_gen.executor.Executor',
        'SchemaGen',
        '--stats',
        """[{"artifact_type": {"name": "ExampleStatisticsPath", "properties": {"state": "STRING", "split": "STRING", "span": "INT", "name": "STRING", "type_name": "STRING"}}, "artifact": {"uri": "/tfx-src/tfx/components/testdata/statistics_gen/train/", "properties": {"split": {"stringValue": "train"}, "type_name": {"stringValue": "ExampleStatisticsPath"}, "span": {"intValue": "0"}}}}, {"artifact_type": {"name": "ExampleStatisticsPath", "properties": {"state": "STRING", "split": "STRING", "type_name": "STRING", "name": "STRING", "span": "INT"}}, "artifact": {"uri": "/tfx-src/tfx/components/testdata/statistics_gen/eval/", "properties": {"split": {"stringValue": "eval"}, "type_name": {"stringValue": "ExampleStatisticsPath"}, "span": {"intValue": "0"}}}}]""",
    ]

    self._client.containers.run(
        image=self._image_tag,
        environment=['WORKFLOW_ID=TEST_WORKFLOW'],
        entrypoint='python',
        volumes={
            self._output_basedir: {
                'bind': self._output_basedir,
                'mode': 'rw'
            }
        },
        command=command)
    self.assertTrue(tf.gfile.ListDirectory(self._output_basedir))


if __name__ == '__main__':
  unittest.main()
