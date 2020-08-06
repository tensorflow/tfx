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
"""BuildSpec helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Optional, Text

import click
from tfx.tools.cli.container_builder import labels
import yaml


class BuildSpec(object):
  """Build specification.

  BuildSpec generates a default build spec if it does not exist.

  Attributes:
    filename: build spec filename.
    build_context: build working directory.
    target_image: target image with no tag.
    target_image_tag: tag of the target image.
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
                   target_image: Optional[Text] = None,
                   build_context: Optional[Text] = None,
                   dockerfile_name: Optional[Text] = None):
    """Generate a default build spec yaml.

    Args:
      filename: build spec filename.
      target_image: target image path. If it contains the tag, the build spec
        will also include it; If it does not, the build spec will tag it as
        'lastest'.
      build_context: local build context path.
      dockerfile_name: dockerfile filename in the build_context.

    Returns:
      BuildSpec instance.
    """
    if os.path.exists(filename):
      raise ValueError('BuildSpec: build spec file %s already exists.' %
                       filename)
    if target_image is None:
      raise ValueError('BuildSpec: target_image is not given.')

    target_image_fields = target_image.split(':')
    if len(target_image_fields) > 2:
      raise ValueError('BuildSpec: target_image is in illegal form.')
    target_image_with_no_tag = target_image_fields[0]
    target_image_tag = 'latest' if len(
        target_image_fields) <= 1 else target_image_fields[1]

    build_context = build_context or labels.BUILD_CONTEXT
    dockerfile_name = dockerfile_name or labels.DOCKERFILE_NAME

    build_spec = {
        'apiVersion': labels.SKAFFOLD_API_VERSION,
        'kind': 'Config',
        'build': {
            'tagPolicy': {
                'envTemplate': {
                    'template': target_image_tag
                }
            },
            'artifacts': [{
                'image': target_image_with_no_tag,
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
      self._target_image = self._buildspec['build']['artifacts'][0]['image']
      self._target_image_tag = self._buildspec['build']['tagPolicy'][
          'envTemplate']['template']
      # For compatibility with old build files which have `{{.IMAGE_NAME}}:tag`
      # format.
      if self._target_image_tag.startswith('{{.IMAGE_NAME}}:'):
        self._target_image_tag = self._target_image_tag.split(':', 2)[-1]

  @property
  def filename(self):
    return self._filename

  @property
  def build_context(self):
    return self._build_context

  @property
  def target_image(self):
    return self._target_image

  @property
  def target_image_tag(self):
    return self._target_image_tag
