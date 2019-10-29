# Lint as: python3
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
"""Tests for tfx.orchestration.config.docker_component_config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.orchestration.config import docker_component_config


class DockerComponentConfigTest(tf.test.TestCase):

  def testToRunArgs(self):
    docker_config = docker_component_config.DockerComponentConfig(
        docker_server_url='http://mock.docker.server',
        environment={'name': 'value'},
        privileged=True,
        volumes=['/local/etc:/local/etc'],
        ports={'2222/tcp': 3333})

    run_args = docker_config.to_run_args()

    self.assertEqual('http://mock.docker.server',
                     docker_config.docker_server_url)
    self.assertDictEqual({'name': 'value'}, run_args['environment'])
    self.assertTrue(run_args['privileged'])
    self.assertListEqual(['/local/etc:/local/etc'], run_args['volumes'])
    self.assertDictEqual({'2222/tcp': 3333}, run_args['ports'])


if __name__ == '__main__':
  tf.test.main()
