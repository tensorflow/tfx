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
"""ContainerBuilder builds the container image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Optional, Text

import click

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
               base_image: Optional[Text] = None,
               skaffold_cmd: Optional[Text] = None,
               buildspec_filename: Optional[Text] = None,
               dockerfile_name: Optional[Text] = None,
               setup_py_filename: Optional[Text] = None):
    """Initialization.

    Args:
      target_image: the target image path to be built.
      base_image: the image path to use as the base image.
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
    self._skaffold_cmd = skaffold_cmd or labels.SKAFFOLD_COMMAND
    buildspec_filename = buildspec_filename or labels.BUILD_SPEC_FILENAME
    dockerfile_name = dockerfile_name or labels.DOCKERFILE_NAME

    if os.path.exists(buildspec_filename):
      self._buildspec = buildspec.BuildSpec(filename=buildspec_filename)
      if target_image is not None:
        click.echo(
            'Target image %s is not used. If the build spec is '
            'provided, update the target image in the build spec '
            'file %s.' % (target_image, buildspec_filename))
    else:
      self._buildspec = buildspec.BuildSpec.load_default(
          filename=buildspec_filename,
          target_image=target_image,
          dockerfile_name=dockerfile_name)

    Dockerfile(
        filename=os.path.join(self._buildspec.build_context, dockerfile_name),
        setup_py_filename=setup_py_filename,
        base_image=base_image)

  def build(self):
    """Build the container and return the built image path with SHA."""
    skaffold_cli = SkaffoldCli(cmd=self._skaffold_cmd)
    image_sha = skaffold_cli.build(self._buildspec)
    target_image = self._buildspec.target_image
    return target_image + '@' + image_sha
