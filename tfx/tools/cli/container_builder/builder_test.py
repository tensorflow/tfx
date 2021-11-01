# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.tools.cli.container_builder.builder."""

from unittest import mock

import docker

import tensorflow as tf
from tfx.tools.cli.container_builder import builder
from tfx.tools.cli.container_builder import dockerfile


class BuilderTest(tf.test.TestCase):

  def testGetImageRepo(self):
    self.assertEqual('tensorflow/tfx',
                     builder._get_image_repo('tensorflow/tfx:latest'))
    self.assertEqual('tensorflow/tfx',
                     builder._get_image_repo('tensorflow/tfx'))
    with self.assertRaises(ValueError):
      builder._get_image_repo('a/b:latest:wrong')

  @mock.patch.object(dockerfile, 'Dockerfile', autospec=True)
  @mock.patch.object(docker, 'APIClient', autospec=True)
  @mock.patch.object(docker, 'from_env', autospec=True)
  def testBuild(self, mock_docker_client, mock_docker_low_client,
                mock_dockerfile):
    mock_build_fn = mock_docker_low_client.return_value.build
    mock_build_fn.return_value = [{'stream': 'foo'}, {'baz': ''}]
    mock_push_fn = mock_docker_client.return_value.images.push
    mock_build_fn.return_value = [{'status': 'bar'}]
    mock_get_registry_data_fn = (
        mock_docker_client.return_value.images.get_registry_data)
    mock_get_registry_data_fn.return_value = mock.MagicMock()
    mock_get_registry_data_fn.return_value.id = 'sha256:01234'

    target_image = 'gcr.io/test/myimage:mytag'
    built_image = builder.build(target_image, 'some:base', 'Dockerfile.test',
                                'setup.py')
    mock_dockerfile.assert_called_once_with('Dockerfile.test', 'setup.py',
                                            'some:base')
    mock_build_fn.assert_called_once()
    mock_push_fn.assert_called_once()
    mock_get_registry_data_fn.assert_called_once_with(target_image)
    self.assertEqual(built_image, 'gcr.io/test/myimage@sha256:01234')


if __name__ == '__main__':
  tf.test.main()
