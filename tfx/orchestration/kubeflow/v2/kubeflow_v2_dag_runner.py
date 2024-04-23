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
"""V2 Kubeflow DAG Runner."""

import datetime
import json
import os
from typing import Any, Dict, List, Optional, Union, MutableMapping
from absl import logging

from kfp.pipeline_spec import pipeline_spec_pb2
from tfx import version
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_node
from tfx.dsl.io import fileio
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.kubeflow.v2 import pipeline_builder
from tfx.utils import telemetry_utils
from tfx.utils import version_utils

from google.protobuf import json_format

KUBEFLOW_TFX_CMD = (
    'python', '-m',
    'tfx.orchestration.kubeflow.v2.container.kubeflow_v2_run_executor')

# If the default_image is set to be a map, the value of this key is used for the
# components whose images are not specified. If not specified, this key will
# have the value of default TFX container image.
_DEFAULT_IMAGE_PATH_KEY = pipeline_builder.DEFAULT_IMAGE_PATH_KEY

# Current schema version for the API proto.
_SCHEMA_VERSION = '2.0.0'

# Default TFX container image/commands to use in KubeflowV2DagRunner.
_KUBEFLOW_TFX_IMAGE = 'gcr.io/tfx-oss-public/tfx:{}'.format(
    version_utils.get_image_version())


def _get_current_time():
  """Gets the current timestamp."""
  return datetime.datetime.now()


class KubeflowV2DagRunnerConfig(pipeline_config.PipelineConfig):
  """Runtime configuration specific to execution on Kubeflow V2 pipelines."""

  def __init__(
      self,
      display_name: Optional[str] = None,
      default_image: Optional[Union[str, MutableMapping[str, str]]] = None,
      default_commands: Optional[List[str]] = None,
      **kwargs
  ):
    """Constructs a Kubeflow V2 runner config.

    Args:
      display_name: Optional human-readable pipeline name. Defaults to the
        pipeline name passed into `KubeflowV2DagRunner.run()`.
      default_image: The default TFX image to be used if not overriden by per
        component specification. It can be a map whose key is a component id and
        value is an image path to set the image by a component level.
      default_commands: Optionally specifies the commands of the provided
        container image. When not provided, the default `ENTRYPOINT` specified
        in the docker image is used. Note: the commands here refers to the K8S
        container command, which maps to Docker entrypoint field. If one
        supplies command but no args are provided for the container, the
        container will be invoked with the provided command, ignoring the
        `ENTRYPOINT` and `CMD` defined in the Dockerfile. One can find more
        details regarding the difference between K8S and Docker conventions at
        https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#notes
      **kwargs: Additional args passed to base PipelineConfig.
    """
    super().__init__(**kwargs)
    self.display_name = display_name
    self.default_image = default_image or _KUBEFLOW_TFX_IMAGE
    if (
        isinstance(self.default_image, MutableMapping)
        and self.default_image.get(_DEFAULT_IMAGE_PATH_KEY) is None
    ):
      self.default_image[_DEFAULT_IMAGE_PATH_KEY] = _KUBEFLOW_TFX_IMAGE
    if default_commands is None:
      self.default_commands = KUBEFLOW_TFX_CMD
    else:
      self.default_commands = default_commands


class KubeflowV2DagRunner(tfx_runner.TfxRunner):
  """Kubeflow V2 pipeline runner (currently for managed pipelines).

  Builds a pipeline job spec in json format based on TFX pipeline DSL object.
  """

  def __init__(self,
               config: KubeflowV2DagRunnerConfig,
               output_dir: Optional[str] = None,
               output_filename: Optional[str] = None):
    """Constructs an KubeflowV2DagRunner for compiling pipelines.

    Args:
      config: An KubeflowV2DagRunnerConfig object to specify runtime
        configuration when running the pipeline in Kubeflow.
      output_dir: An optional output directory into which to output the pipeline
        definition files. Defaults to the current working directory.
      output_filename: An optional output file name for the pipeline definition
        file. The file output format will be a JSON-serialized PipelineJob pb
        message. Defaults to 'pipeline.json'.
    """
    if not isinstance(config, KubeflowV2DagRunnerConfig):
      raise TypeError('config must be type of KubeflowV2DagRunnerConfig.')
    super().__init__()
    self._config = config
    self._output_dir = output_dir or os.getcwd()
    self._output_filename = output_filename or 'pipeline.json'
    self._exit_handler = None

  def set_exit_handler(self, exit_handler: base_node.BaseNode):
    """Set exit handler components for the Kubeflow V2(Vertex AI) dag runner.

    This feature is currently experimental without backward compatibility
    gaurantee.

    Args:
      exit_handler: exit handler component.
    """
    if not exit_handler:
      logging.error('Setting empty exit handler is not allowed.')
      return
    self._exit_handler = exit_handler

  def run(self,
          pipeline: tfx_pipeline.Pipeline,
          parameter_values: Optional[Dict[str, Any]] = None,
          write_out: Optional[bool] = True) -> Dict[str, Any]:
    """Compiles a pipeline DSL object into pipeline file.

    Args:
      pipeline: TFX pipeline object.
      parameter_values: mapping from runtime parameter names to its values.
      write_out: set to True to actually write out the file to the place
        designated by output_dir and output_filename. Otherwise return the
        JSON-serialized pipeline job spec.

    Returns:
      Returns the JSON pipeline job spec.

    Raises:
      RuntimeError: if trying to write out to a place occupied by an existing
      file.
    """
    for component in pipeline.components:
      # TODO(b/187122662): Pass through pip dependencies as a first-class
      # component flag.
      if isinstance(component, base_component.BaseComponent):
        component._resolve_pip_dependencies(  # pylint: disable=protected-access
            pipeline.pipeline_info.pipeline_root)

    # TODO(b/166343606): Support user-provided labels.
    # TODO(b/169095387): Deprecate .run() method in favor of the unified API
    # client.
    display_name = (
        self._config.display_name or pipeline.pipeline_info.pipeline_name)
    pipeline_spec = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline,
        default_image=self._config.default_image,
        default_commands=self._config.default_commands,
        exit_handler=self._exit_handler).build()
    pipeline_spec.sdk_version = 'tfx-{}'.format(version.__version__)
    pipeline_spec.schema_version = _SCHEMA_VERSION
    runtime_config = pipeline_builder.RuntimeConfigBuilder(
        pipeline_info=pipeline.pipeline_info,
        parameter_values=parameter_values).build()
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'kubeflow_v2'}):
      result = pipeline_spec_pb2.PipelineJob(
          display_name=display_name or pipeline.pipeline_info.pipeline_name,
          labels=telemetry_utils.make_labels_dict(),
          runtime_config=runtime_config)
    result.pipeline_spec.update(json_format.MessageToDict(pipeline_spec))
    pipeline_json_dict = json_format.MessageToDict(result)
    if write_out:
      if fileio.exists(self._output_dir) and not fileio.isdir(self._output_dir):
        raise RuntimeError('Output path: %s is pointed to a file.' %
                           self._output_dir)
      if not fileio.exists(self._output_dir):
        fileio.makedirs(self._output_dir)

      with fileio.open(
          os.path.join(self._output_dir, self._output_filename), 'wb') as f:
        f.write(json.dumps(pipeline_json_dict, sort_keys=True))

    return pipeline_json_dict
