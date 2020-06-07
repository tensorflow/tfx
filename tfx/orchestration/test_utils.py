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
"""Common utilities for testing various runners."""

import datetime
import os
import random
import string
import subprocess
import time

from absl import logging
import docker
import tensorflow as tf

from google.cloud import storage


def random_id():
  """Generates a random string that is also a valid Kubernetes DNS name."""
  random.seed(datetime.datetime.now())

  choices = string.ascii_lowercase + string.digits
  return '{}-{}'.format(datetime.datetime.now().strftime('%s'),
                        ''.join([random.choice(choices) for _ in range(10)]))


class Timer:
  """Helper class to time operations in pipeline e2e tests."""

  def __init__(self, operation: str):
    """Creates a context object to measure time taken.

    Args:
      operation: A description of the operation being measured.
    """
    self._operation = operation

  def __enter__(self):
    self._start = time.time()

  def __exit__(self, *unused_args):
    self._end = time.time()

    logging.info(
        'Timing Info >> Operation: %s Elapsed time in seconds: %d',
        self._operation, self._end - self._start)


class BasePipelineE2ETest(tf.test.TestCase):
  """Base class that defines testing harness for different runners."""

  @classmethod
  def setUpClass(cls):
    super(BasePipelineE2ETest, cls).setUpClass()

    # Create a container image for use by test pipelines.
    cls._CONTAINER_IMAGE = '{}:{}'.format(cls._BASE_CONTAINER_IMAGE,
                                          random_id())
    cls._build_and_push_docker_image(cls._CONTAINER_IMAGE)

  @classmethod
  def tearDownClass(cls):
    super(BasePipelineE2ETest, cls).tearDownClass()

    # Delete container image used in tests.
    logging.info('Deleting image %s', cls._CONTAINER_IMAGE)
    subprocess.run(
        ['gcloud', 'container', 'images', 'delete', cls._CONTAINER_IMAGE],
        check=True)

  @classmethod
  def _build_and_push_docker_image(cls, container_image: str):
    client = docker.from_env()

    logging.info('Building image %s', container_image)
    with Timer('BuildingTFXContainerImage'):
      _ = client.images.build(
          path=cls._REPO_BASE,
          dockerfile='tfx/tools/docker/Dockerfile',
          tag=container_image,
          buildargs={
              # Skip license gathering for tests.
              'gather_third_party_licenses': 'false',
          },
      )

    logging.info('Pushing image %s', container_image)
    with Timer('PushingTFXContainerImage'):
      client.images.push(repository=container_image)

  def _delete_test_dir(self, test_id: str):
    """Deletes files for this test including the module file and data files.

    Args:
      test_id: Randomly generated id of the test.
    """
    self._delete_gcs_files('test_data/{}'.format(test_id))

  def _delete_gcs_files(self, path: str):
    """Deletes files under specified path in the test bucket.

    Args:
      path: path(or prefix) of the file to delete.
    """
    client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = client.get_bucket(self._BUCKET_NAME)
    logging.info('Deleting files under GCS bucket path: %s', path)

    with Timer('ListingAndDeletingFilesFromGCS'):
      blobs = bucket.list_blobs(prefix=path)
      bucket.delete_blobs(blobs)

  def _delete_pipeline_output(self, pipeline_name: str):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    self._delete_gcs_files('test_output/{}'.format(pipeline_name))

  def _pipeline_root(self, pipeline_name: str):
    """Generates the pipeline root path based on pipeline name."""
    return os.path.join(self._test_output_dir, pipeline_name)
