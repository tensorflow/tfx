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
"""Dockerfile helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text, Optional

import click

from tfx import version
from tfx.tools.cli.container_builder import labels


_DEFAULT_DOCKERFILE_CONTENT_WITH_SETUP_PY = '''FROM %s
WORKDIR /pipeline
COPY ./ ./
RUN python3 %s install'''

_DEFAULT_DOCKERFILE_CONTENT_WITHOUT_SETUP_PY = '''FROM %s
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"'''


class Dockerfile(object):
  """Dockerfile class.

  Dockerfile generates a default dockerfile if it does not exist.

  Attributes:
    filename: dockerfile filename.
    setup_py_filename: setup.py filename that defines the pipeline PIP package.
  """

  def __init__(self,
               filename: Text,
               setup_py_filename: Optional[Text] = None,
               base_image: Optional[Text] = None):
    self.filename = filename
    if os.path.exists(self.filename):
      return

    if base_image is None and version.__version__.endswith('.dev'):
      raise ValueError('Cannot find a base image automatically in development /'
                       ' nightly version. Please specify a base image using'
                       ' --build-base-image flag.')

    base_image = base_image or labels.BASE_IMAGE
    setup_py_filename = setup_py_filename or labels.SETUP_PY_FILENAME
    if os.path.exists(setup_py_filename):
      click.echo('Generating Dockerfile with python package installation '
                 'based on %s.' % setup_py_filename)
      self._generate_default(_DEFAULT_DOCKERFILE_CONTENT_WITH_SETUP_PY %
                             (base_image, setup_py_filename))
      return

    click.echo('No local %s, copying the directory and '
               'configuring the PYTHONPATH.' % setup_py_filename)
    self._generate_default(_DEFAULT_DOCKERFILE_CONTENT_WITHOUT_SETUP_PY %
                           base_image)

  def _generate_default(self, contents):
    """Generate a dockerfile with the contents."""
    with open(self.filename, 'w+') as f:
      f.write(contents)
