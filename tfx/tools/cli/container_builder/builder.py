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
"""ContainerBuilder builds the container image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import click
from typing import Optional, Text

from tfx.tools.cli.container_builder import buildspec
from tfx.tools.cli.container_builder import labels
from tfx.tools.cli.container_builder.dockerfile import Dockerfile
from tfx.tools.cli.container_builder.skaffold_cli import SkaffoldCli


# TODO(b/142357382): add e2e tests.
class ContainerBuilder(object):
  """Build containers.

  ContainerBuilder prepares the build files and run Skaffold to build the
  containers.

  Attributes:
    _buildspec: BuildSpec instance.
    _skaffold_cmd: Skaffold command.
  """

  def __init__(self,
               target_image: Optional[Text] = None,
               skaffold_cmd: Optional[Text] = labels.SKAFFOLD_COMMAND,
               buildspec_filename: Optional[Text] = labels.BUILD_SPEC_FILENAME,
               dockerfile_name: Optional[Text] = labels.DOCKERFILE_NAME,
               setup_py_filename: Optional[Text] = labels.SETUP_PY_FILENAME):
    """Initialization.

    Args:
      target_image: the target image path to be built.
      skaffold_cmd: skaffold command.
      buildspec_filename: the buildspec file path that is accessible to the
        current execution environment. It could be either absolute path or
        relative path.
      dockerfile_name: the dockerfile name, which is stored in the workspace
        directory. The workspace directory is specified in the build spec and
        the default workspace directory is '.'.
      setup_py_filename: the setup.py file name, which is used to build a
        python package for the workspace directory. If not specified, the
        whole directory is copied and PYTHONPATH is configured.
    """
    self._skaffold_cmd = skaffold_cmd
    if os.path.exists(buildspec_filename):
      self._buildspec = buildspec.BuildSpec(filename=buildspec_filename)
      if target_image is not None:
        click.echo(
            'Target image %s is not used. If the build spec is '
            'provicded, update the target image in the build spec '
            'file %s.' % (target_image, buildspec_filename))
    else:
      self._buildspec = buildspec.BuildSpec.load_default(
          filename=buildspec_filename,
          target_image=target_image,
          dockerfile_name=dockerfile_name)

    Dockerfile(
        filename=os.path.join(self._buildspec.build_context, dockerfile_name),
        setup_py_filename=setup_py_filename)

  def build(self):
    """Build the container."""
    click.echo('Use skaffold to build the container image.')
    skaffold_cli = SkaffoldCli(cmd=self._skaffold_cmd)
    skaffold_cli.build(buildspec_filename=self._buildspec.filename)
