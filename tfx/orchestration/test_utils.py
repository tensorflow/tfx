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
import random
import string
import time

from absl import logging
import docker

from google.cloud import storage


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


def random_id() -> str:
  """Generates a random string that is also a valid Kubernetes DNS name.

  Returns:
    A random string valid for Kubernetes DNS name.
  """
  random.seed(datetime.datetime.now())

  choices = string.ascii_lowercase + string.digits
  return '{}-{}'.format(datetime.datetime.now().strftime('%s'),
                        ''.join([random.choice(choices) for _ in range(10)]))


def build_and_push_docker_image(container_image: str, repo_base: str):
  """Build and push docker image using `tfx/tools/docker/Dockerfile`.

  Args:
    container_image: Docker container image name.
    repo_base: The src path to use to build docker image.
  """
  client = docker.from_env()

  logging.info('Building image %s', container_image)
  with Timer('BuildingTFXContainerImage'):
    _ = client.images.build(
        path=repo_base,
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


def delete_gcs_files(gcp_project_id: str, bucket_name: str, path: str):
  """Deletes files under specified path in the test bucket.

  Args:
    gcp_project_id: GCP project ID.
    bucket_name: GCS bucket name.
    path: path(or prefix) of the file to delete.
  """
  client = storage.Client(project=gcp_project_id)
  bucket = client.get_bucket(bucket_name)
  logging.info('Deleting files under GCS bucket path: %s', path)

  with Timer('ListingAndDeletingFilesFromGCS'):
    blobs = list(bucket.list_blobs(prefix=path))
    bucket.delete_blobs(blobs)
