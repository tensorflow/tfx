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
"""BuildSpec helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import click
from typing import Text
import yaml

from tfx.tools.cli.container_builder import labels


class BuildSpec(object):
  """Build specification.

  BuildSpec generates a default build spec if it does not exist.

  Attributes:
    filename: build spec filename.
    build_context: build working directory.
    _buildspec: in-memory representation of the build spec.
  """

  def __init__(self,
               filename: Text = labels.BUILD_SPEC_FILENAME):
    self._filename = filename
    if not os.path.exists(self._filename):
      raise ValueError('BuildSpec:: build spec file %s does not exist.' %
                       filename)
    self._read_existing_build_spec()

  @staticmethod
  def load_default(filename: Text = labels.BUILD_SPEC_FILENAME,
                   target_image: Text = None,
                   build_context: Text = labels.BUILD_CONTEXT,
                   dockerfile_name: Text = labels.DOCKERFILE_NAME):
    """Generate a default build spec yaml."""
    if os.path.exists(filename):
      raise ValueError('BuildSpec: build spec file %s already exists.' %
                       filename)
    if target_image is None:
      raise ValueError('BuildSpec: target_image is not given.')
    build_spec = {
        'apiVersion': labels.SKAFFOLD_API_VERSION,
        'kind': 'Config',
        'build': {
            'artifacts': [{
                'image': target_image,
                'context': build_context,
                'docker': {
                    'dockerfile': dockerfile_name
                }
            }]
        }
    }
    with open(filename, 'w') as f:
      yaml.dump(build_spec, f)
    return BuildSpec(filename)

  def _read_existing_build_spec(self):
    """Read existing build spec yaml."""
    with open(self.filename, 'r') as f:
      click.echo('Reading build spec from %s' % self.filename)
      self._buildspec = yaml.safe_load(f)
      if len(self._buildspec['build']['artifacts']) != 1:
        raise RuntimeError('The build spec contains multiple artifacts however'
                           'only one is supported.')
      self._build_context = self._buildspec['build']['artifacts'][0]['context']

  @property
  def filename(self):
    return self._filename

  @property
  def build_context(self):
    return self._build_context
