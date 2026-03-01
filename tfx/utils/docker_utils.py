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
"""Utilities to manage docker container images."""

import subprocess

import docker


def delete_image(name: str, remote: bool = True):
  """Delete container image in local and remote registry."""
  client = docker.from_env()

  # List all images including un-tagged images and delete all of them.
  images = client.images.list(name)
  # Sort list to delete children images first.
  images.sort(key=lambda image: image.attrs['Created'], reverse=True)
  for image in images:
    client.images.remove(image.id, force=True)
    if remote:
      # NOTE: RepoDigest != id. Use repo digests when deleting remote images.
      remote_image = image.attrs['RepoDigests'][0]
      subprocess.check_output([
          'gcloud', 'container', 'images', 'delete', remote_image, '--quiet',
          '--force-delete-tags'
      ])
  client.close()
