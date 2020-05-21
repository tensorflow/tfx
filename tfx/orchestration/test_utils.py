# Lint as: python3
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
"""Common utility for testing various orchestrators."""

import datetime
import os
import random
import string
import subprocess
import time
from typing import Text

import absl
import docker
import tensorflow as tf

from google.cloud import storage


class _Timer(object):
  """Helper class to time operations in Kubeflow e2e tests."""

  def __init__(self, operation: Text):
    """Creates a context object to measure time taken.

    Args:
      operation: A description of the operation being measured.
    """
    self._operation = operation

  def __enter__(self):
    self._start = time.time()

  def __exit__(self, *unused_args):
    self._end = time.time()

    absl.logging.info(
        'Timing Info >> Operation: %s Elapsed time in seconds: %d' %
        (self._operation, self._end - self._start))


class BasePipelineE2ETest(tf.test.TestCase):
  """Base class that defines testing harness for different pipelines."""

  @classmethod
  def setUpClass(cls):
    super(BasePipelineE2ETest, cls).setUpClass()

    # Create a container image for use by test pipelines.
    cls._container_image = '{}:{}'.format(cls._base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(BasePipelineE2ETest, cls).tearDownClass()

    # Delete container image used in tests.
    absl.logging.info('Deleting image {}'.format(cls._container_image))
    subprocess.run(
        ['gcloud', 'container', 'images', 'delete', cls._container_image],
        check=True)

  @classmethod
  def _build_and_push_docker_image(cls, container_image: Text):
    client = docker.from_env()

    absl.logging.info('Building image {}'.format(container_image))
    with _Timer('BuildingTFXContainerImage'):
      _ = client.images.build(
          path=cls._repo_base,
          dockerfile='tfx/tools/docker/Dockerfile',
          tag=container_image,
          buildargs={
              # Skip license gathering for tests.
              'gather_third_party_licenses': 'false',
          },
      )

    absl.logging.info('Pushing image {}'.format(container_image))
    with _Timer('PushingTFXContainerImage'):
      client.images.push(repository=container_image)

  @staticmethod
  def _random_id():
    """Generates a random string that is also a valid Kubernetes DNS name."""
    random.seed(datetime.datetime.now())

    choices = string.ascii_lowercase + string.digits
    return '{}-{}'.format(datetime.datetime.now().strftime('%s'),
                          ''.join([random.choice(choices) for _ in range(10)]))

  def _pipeline_root(self, pipeline_name: Text):
    """Generates the pipeline root path based on pipeline name."""
    return os.path.join(self._test_output_dir, pipeline_name)

  def _delete_test_dir(self, test_id: Text):
    """Deletes files for this test including the module file and data files.

    Args:
      test_id: Randomly generated id of the test.
    """
    self._delete_gcs_files('test_data/{}'.format(test_id))

  def _delete_gcs_files(self, path: Text):
    """Deletes files under specified path in the test bucket.

    Args:
      path: path(or prefix) of the file to delete.
    """
    client = storage.Client(project=self._gcp_project_id)
    bucket = client.get_bucket(self._bucket_name)
    absl.logging.info('Deleting files under GCS bucket path: {}'.format(path))

    with _Timer('ListingAndDeletingFilesFromGCS'):
      blobs = bucket.list_blobs(prefix=path)
      bucket.delete_blobs(blobs)

  def _delete_pipeline_output(self, pipeline_name: Text):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    self._delete_gcs_files('test_output/{}'.format(pipeline_name))
