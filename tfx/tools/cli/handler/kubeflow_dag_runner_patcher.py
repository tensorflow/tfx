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
"""Patches KubeflowDagRunner to read and update argument during compilation."""

import os
import tempfile
import typing
from typing import Any, Callable, MutableMapping, Optional, Type

from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.tools.cli.handler import dag_runner_patcher


def _get_temporary_package_filename(pipeline_name: str, directory: str) -> str:
  # mkstemp will create and open a file named 'temp_xxxxx.tar.gz'.
  fd, path = tempfile.mkstemp('.tar.gz', f'temp_{pipeline_name}', directory)
  os.close(fd)
  return os.path.basename(path)


class KubeflowDagRunnerPatcher(dag_runner_patcher.DagRunnerPatcher):
  """Patches KubeflowDagRunner.run() with several customizations for CLI."""

  USE_TEMPORARY_OUTPUT_FILE = 'use_temporary_output_file'
  OUTPUT_FILE_PATH = 'output_file_path'

  def __init__(self,
               call_real_run: bool,
               use_temporary_output_file: bool = False,
               build_image_fn: Optional[Callable[[str], str]] = None):
    """Initialize KubeflowDagRunnerPatcher.

    Args:
      call_real_run: Specify KubeflowDagRunner.run() should be called.
      use_temporary_output_file: If True, we will override the default value of
        the pipeline package output path. Even if it is set to True, if users
        specified a path in KubeflowDagRunner then this option will be ignored.
      build_image_fn: If specified, call the function with the configured
        tfx_image in the pipeline. The result of the function will be
        substituted as a new tfx_image of the pipeline.
    """
    super().__init__(call_real_run)
    self._build_image_fn = build_image_fn
    self._use_temporary_output_file = use_temporary_output_file

  def _before_run(self, runner: tfx_runner.TfxRunner,
                  pipeline: tfx_pipeline.Pipeline,
                  context: MutableMapping[str, Any]) -> None:
    runner = typing.cast(kubeflow_dag_runner.KubeflowDagRunner, runner)
    runner_config = typing.cast(kubeflow_dag_runner.KubeflowDagRunnerConfig,
                                runner.config)
    if self._build_image_fn is not None:
      # Replace the image for the pipeline with the newly built image name.
      # This new image name will include the sha256 image id.
      runner_config.tfx_image = self._build_image_fn(runner_config.tfx_image)

    # pylint: disable=protected-access
    context[self.USE_TEMPORARY_OUTPUT_FILE] = (
        runner._output_filename is None and self._use_temporary_output_file)
    if context[self.USE_TEMPORARY_OUTPUT_FILE]:
      # Replace the output of the kfp compile to a temporary file.
      # This file will be deleted after job submission in kubeflow_handler.py
      runner._output_filename = _get_temporary_package_filename(
          context[self.PIPELINE_NAME], runner._output_dir)
    output_filename = (
        runner._output_filename or
        kubeflow_dag_runner.get_default_output_filename(
            context[self.PIPELINE_NAME]))
    context[self.OUTPUT_FILE_PATH] = os.path.join(runner._output_dir,
                                                  output_filename)

  def get_runner_class(self) -> Type[tfx_runner.TfxRunner]:
    return kubeflow_dag_runner.KubeflowDagRunner
