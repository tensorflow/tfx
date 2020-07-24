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

import json
import os
import re
import subprocess
from typing import Text

import click

from tfx.tools.cli.container_builder import labels


class SkaffoldCli(object):
  """Skaffold CLI."""

  def __init__(self, cmd=labels.SKAFFOLD_COMMAND):
    self._cmd = cmd or labels.SKAFFOLD_COMMAND
    try:
      subprocess.run(['which', self._cmd], check=True)
    except subprocess.CalledProcessError:
      click.echo('No executable %s' % self._cmd)
      click.echo('please refer to '
                 'https://github.com/GoogleContainerTools/skaffold/releases '
                 'for installation instructions.')
      raise RuntimeError

  def build(self, buildspec_filename: Text = labels.BUILD_SPEC_FILENAME):
    """Builds an image and return the image SHA."""
    if not os.path.exists(buildspec_filename):
      raise ValueError('Build spec: %s does not exist.' % buildspec_filename)
    output = subprocess.check_output([
        self._cmd, 'build', '-q', '--output={{json .}}', '-f',
        buildspec_filename
    ]).decode('utf-8')
    full_image_name_with_tag = json.loads(output)['builds'][0]['tag']
    m = re.search(r'sha256:[0-9a-f]{64}', full_image_name_with_tag)
    if m is None:
      raise RuntimeError('SkaffoldCli: built image SHA is not found.')
    return m.group(0)
