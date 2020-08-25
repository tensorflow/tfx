# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Definition of Asynchronous TFX runner."""

import os
from typing import Any, Iterable

from absl import logging
from multiprocessing import Process
from tfx.orchestration import metadata
from tfx.orchestration.portable import async_launcher
from tfx.orchestration.portable import tfx_runner
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import telemetry_utils

from ml_metadata.proto import metadata_store_pb2


class MultipleComponentsRunner(tfx_runner.TfxRunner):
  """Asynchronous Tfx runner."""

  def __init__(self):
    """Initializes MultipleComponentsRunner as a TFX orchestrator.
    """

  def _launch_component_async(self, 
                              pipeline_node: pipeline_pb2.PipelineNode,
                              mlmd_connection: metadata.Metadata,
                              pipeline_info: pipeline_pb2.PipelineInfo,
                              pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec):

    async_launcher = async_launcher.Launcher(
        pipeline_node=pipeline_node,
        mlmd_connection=mlmd_connection,
        pipeline_info=pipeline_info,
        pipeline_runtime_spec=pipeline_runtime_spec)

    _component_id = pipeline_node.node_info.id

    async_launcher.launch()
    logging.info('Component %s is running.', _component_id)


  def run(self, pipeline: pipeline_pb2.Pipeline) -> None:
    """Deploys given pipeline asynchronously using processes.

    Args:
      pipeline: Logical pipeline in IR format.
    """
    # For CLI, while creating or updating pipeline, pipeline_args are extracted
    # and hence we avoid deploying the pipeline.
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      return

    # TODO(b/163003901): MLMD connection config should be passed in via IR.
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    for node in pipeline.nodes:
      pipeline_node = node.pipeline_node
      pipeline_info = pipeline.pipeline_info
      runtime_spec = pipeline.runtime_spec

      new_process = Process(target=self._launch_component_async,
                            args=(pipeline_node, mlmd_connection,
                                  pipeline_info, runtime_spec))
      new_process.start()

    

