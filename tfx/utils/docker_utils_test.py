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
"""Tests for tfx.utils.docker_utils."""

import subprocess
from unittest import mock

import docker

import tensorflow as tf
from tfx.utils import docker_utils


class DockerUtilsTest(tf.test.TestCase):

  @mock.patch.object(docker, 'from_env', autospec=True)
  @mock.patch.object(subprocess, 'check_output', autospec=True)
  def testDeleteImage(self, mock_check_output, mock_docker):
    image_name = 'gcr.io/some/name'
    mock_client = mock_docker.return_value
    mock_client.images.list.return_value = [
        docker.models.images.Image(
            attrs={
                'Id': 'sha256:abcdefghijkl',
                'Created': '2019-06-03T19:57:07.576509403Z',
                'RepoDigests': ['gcr.io/some/name@sha256:123456789'],
            })
    ]

    docker_utils.delete_image(image_name)

    mock_client.images.list.assert_called_once_with(image_name)
    mock_client.images.remove.assert_called_once_with(
        'sha256:abcdefghijkl', force=True)
    mock_check_output.assert_called_once_with([
        'gcloud', 'container', 'images', 'delete',
        'gcr.io/some/name@sha256:123456789', '--quiet', '--force-delete-tags'
    ])

  @mock.patch.object(docker, 'from_env', autospec=True)
  @mock.patch.object(subprocess, 'check_output', autospec=True)
  def testDeleteImageLocal(self, mock_check_output, mock_docker):
    image_name = 'gcr.io/some/name'
    mock_client = mock_docker.return_value
    mock_client.images.list.return_value = [
        docker.models.images.Image(
            attrs={
                'Id': 'sha256:abcdefghijkl',
                'Created': '2019-06-03T19:57:07.576509403Z',
            })
    ]

    docker_utils.delete_image(image_name, remote=False)

    mock_check_output.assert_not_called()

if __name__ == '__main__':
  tf.test.main()
