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
import click
from typing import Text

from tfx.tools.cli.container_builder import labels


DEFAULT_DOCKERFILE_CONTENT_WITH_SETUP_PY = '''FROM tensorflow/tfx:0.14.0
WORKDIR /pipeline
COPY ./ ./
RUN python3 %s install'''

DEFAULT_DOCKERFILE_CONTENT_WITHOUT_SETUP_PY = '''FROM tensorflow/tfx:0.14.0
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"'''


class Dockerfile(object):
  """Dockerfile class.

  Dockerfile generates a default dockerfile if it does not exist.

  Attributes:
    filename: dockerfile filename.
  """

  def __init__(self,
               filename: Text = labels.DOCKERFILE_NAME,
               setup_py_filename: Text = labels.SETUP_PY_FILENAME):
    self.filename = filename
    if os.path.exists(self.filename):
      return
    if os.path.exists(setup_py_filename):
      click.echo('Generating Dockerfile with python package installation '
                 'based on %s.' % setup_py_filename)
      self._generate_default(DEFAULT_DOCKERFILE_CONTENT_WITH_SETUP_PY %
                             setup_py_filename)
    else:
      click.echo('No local %s, copying the directory and '
                 'configuring the PYTHONPATH.' % setup_py_filename)
      self._generate_default(DEFAULT_DOCKERFILE_CONTENT_WITHOUT_SETUP_PY)

  def _generate_default(self, contents):
    """Generate a dockerfile with the contents."""
    with open(self.filename, 'w+') as f:
      f.write(contents)
