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
"""Patches KubeflowV2DagRunner to read and update argument during compilation."""

import os
import typing
from typing import Any, Callable, MutableMapping, Optional, Type

from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.tools.cli.handler import dag_runner_patcher


OUTPUT_FILENAME = 'pipeline.json'


class KubeflowV2DagRunnerPatcher(dag_runner_patcher.DagRunnerPatcher):
  """Patches KubeflowV2DagRunner.run() with several customizations for CLI."""

  OUTPUT_FILE_PATH = 'output_file_path'

  def __init__(self,
               call_real_run: bool,
               build_image_fn: Optional[Callable[[str], str]] = None,
               prepare_dir_fn: Optional[Callable[[str], str]] = None):
    """Initialize KubeflowV2DagRunnerPatcher.

    Args:
      call_real_run: Specify KubeflowV2DagRunner.run() should be called.
      build_image_fn: If specified, the function will be called before run()
        with the configured tfx_image in the pipeline. The result of the
        function will be substituted as a new tfx_image of the pipeline.
      prepare_dir_fn: If specified, this function will be called with
        pipeline_name as an argument. Before the run() is called.
        The result of the function will be used as a directory to store the
        pipeline.
    """
    super().__init__(call_real_run)
    self._build_image_fn = build_image_fn
    self._prepare_dir_fn = prepare_dir_fn

  def _before_run(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, runner: tfx_runner.TfxRunner, pipeline: tfx_pipeline.Pipeline,
      context: MutableMapping[str, Any]) -> None:
    runner = typing.cast(kubeflow_v2_dag_runner.KubeflowV2DagRunner, runner)
    runner_config = typing.cast(
        kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig, runner.config)
    if self._build_image_fn is not None:
      # Replace the image for the pipeline with the newly built image name.
      # This new image name will include the sha256 image id.
      runner_config.default_image = self._build_image_fn(
          runner_config.default_image)

    # pylint: disable=protected-access
    if self._prepare_dir_fn is not None:
      runner._output_dir = self._prepare_dir_fn(context[self.PIPELINE_NAME])
      runner._output_filename = OUTPUT_FILENAME

    context[self.OUTPUT_FILE_PATH] = os.path.join(runner._output_dir,
                                                  runner._output_filename)

  def get_runner_class(self) -> Type[tfx_runner.TfxRunner]:
    return kubeflow_v2_dag_runner.KubeflowV2DagRunner
