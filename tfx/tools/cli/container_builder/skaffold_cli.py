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
"""Skaffold Cli helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
from typing import Text

import click
import docker

from tfx.tools.cli.container_builder import buildspec
from tfx.tools.cli.container_builder import labels


class SkaffoldCli(object):
  """Skaffold CLI."""

  def __init__(self, cmd: Text = labels.SKAFFOLD_COMMAND):
    self._cmd = cmd or labels.SKAFFOLD_COMMAND
    try:
      subprocess.run(['which', self._cmd],
                     check=True,
                     stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
      click.echo('No executable %s' % self._cmd)
      click.echo('please refer to '
                 'https://github.com/GoogleContainerTools/skaffold/releases '
                 'for installation instructions.')
      raise RuntimeError

  def build(self, spec: buildspec.BuildSpec) -> Text:
    """Builds an image and return the image SHA."""
    if not os.path.exists(spec.filename):
      raise ValueError('Build spec: %s does not exist.' % spec.filename)

    proc = subprocess.Popen([
        self._cmd, 'build', '-f', spec.filename
    ], bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
      click.echo('[Skaffold] %s' % line.decode('utf-8').rstrip())
    proc.communicate()  # wait until the child process exit. No output expected.
    if proc.returncode != 0:
      raise RuntimeError('skaffold failed to build an image with {}.'.format(
          spec.filename))

    docker_client = docker.from_env()
    return docker_client.images.get_registry_data(spec.target_image + ':' +
                                                  spec.target_image_tag).id
