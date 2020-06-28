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
"""Definition of Beam TFX runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from typing import Any, Iterable, List, Optional, Text, Type

import absl
import apache_beam as beam

from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.orchestration.test_pipelines.download_grep_print_pipeline import create_pipeline_component_instances
from tfx.utils import telemetry_utils


# pipeline
#   def __init__(self,
#                pipeline_name: Text,
#                pipeline_root: Text,
#                metadata_connection_config: Optional[
#                    metadata_store_pb2.ConnectionConfig] = None,
#                components: Optional[List[base_node.BaseNode]] = None,
#                enable_cache: Optional[bool] = False,
#                beam_pipeline_args: Optional[List[Text]] = None,
#                **kwargs):
#     """Initialize pipeline.

#     Args:
#       pipeline_name: Name of the pipeline;
#       pipeline_root: Path to root directory of the pipeline;
#       metadata_connection_config: The config to connect to ML metadata.
#       components: A list of components in the pipeline (optional only for
#         backward compatible purpose to be used with deprecated
#         PipelineDecorator).
#       enable_cache: Whether or not cache is enabled for this run.
#       beam_pipeline_args: Pipeline arguments for Beam powered Components.
#       **kwargs: Additional kwargs forwarded as pipeline args.
#     """

# Directory and data locations.  This example assumes all of the flowers
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_name = 'test'
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)

def _create_pipeline() -> pipeline.Pipeline:

    pipeline_name = _pipeline_name
    pipeline_root = _pipeline_root

    components = create_pipeline_component_instances("www.google.com", "google")

    # in cluster connection config
    metadata_connection_config = metadata.mysql_metadata_connection_config(
    host='mysql', port=3306, username='root', database='mysql', password='')

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
        enable_cache=True,
#        beam_pipeline_args=beam_pipeline_args,
    ) 


if __name__ == '__main__':
    # first, create a tfx pipiline
    _pipeline = _create_pipeline()
    # use kubernetes dag runner to run the pipeline
    kubernetes_dag_runner.KubernetesDagRunner().run(tfx_pipeline=_pipeline)