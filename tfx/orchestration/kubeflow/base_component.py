# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kubeflow Pipelines based implementation of TFX components.

These components are lightweight wrappers around the KFP DSL's ContainerOp,
and ensure that the container gets called with the right set of input
arguments. It also ensures that each component exports named output
attributes that are consistent with those provided by the native TFX
components, thus ensuring that both types of pipeline definitions are
compatible.
Note: This requires Kubeflow Pipelines SDK to be installed.
"""

from typing import Dict, List, Set

from absl import logging
from kfp import dsl
from kubernetes import client as k8s_client
from tfx.dsl.components.base import base_node as tfx_base_node
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import utils
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import json_format

# TODO(b/166202742): Consolidate container entrypoint with TFX image's default.
_COMMAND = ['python', '-m', 'tfx.orchestration.kubeflow.container_entrypoint']

_WORKFLOW_ID_KEY = 'WORKFLOW_ID'


def _encode_runtime_parameter(param: data_types.RuntimeParameter) -> str:
  """Encode a runtime parameter into a placeholder for value substitution."""
  if param.ptype is int:
    type_enum = pipeline_pb2.RuntimeParameter.INT
  elif param.ptype is float:
    type_enum = pipeline_pb2.RuntimeParameter.DOUBLE
  else:
    type_enum = pipeline_pb2.RuntimeParameter.STRING
  type_str = pipeline_pb2.RuntimeParameter.Type.Name(type_enum)
  return f'{param.name}={type_str}:{str(dsl.PipelineParam(name=param.name))}'


# TODO(hongyes): renaming the name to KubeflowComponent.
class BaseComponent:
  """Base component for all Kubeflow pipelines TFX components.

  Returns a wrapper around a KFP DSL ContainerOp class, and adds named output
  attributes that match the output names for the corresponding native TFX
  components.
  """

  def __init__(
      self,
      component: tfx_base_node.BaseNode,
      depends_on: Set[dsl.ContainerOp],
      pipeline: tfx_pipeline.Pipeline,
      pipeline_root: dsl.PipelineParam,
      tfx_image: str,
      kubeflow_metadata_config: kubeflow_pb2.KubeflowMetadataConfig,
      tfx_ir: pipeline_pb2.Pipeline, pod_labels_to_attach: Dict[str, str],
      runtime_parameters: List[data_types.RuntimeParameter]):
    """Creates a new Kubeflow-based component.

    This class essentially wraps a dsl.ContainerOp construct in Kubeflow
    Pipelines.

    Args:
      component: The logical TFX component to wrap.
      depends_on: The set of upstream KFP ContainerOp components that this
        component will depend on.
      pipeline: The logical TFX pipeline to which this component belongs.
      pipeline_root: The pipeline root specified, as a dsl.PipelineParam
      tfx_image: The container image to use for this component.
      kubeflow_metadata_config: Configuration settings for connecting to the
        MLMD store in a Kubeflow cluster.
      tfx_ir: The TFX intermedia representation of the pipeline.
      pod_labels_to_attach: Dict of pod labels to attach to the GKE pod.
      runtime_parameters: Runtime parameters of the pipeline.
    """

    utils.replace_placeholder(component)

    arguments = [
        '--pipeline_root',
        pipeline_root,
        '--kubeflow_metadata_config',
        json_format.MessageToJson(
            message=kubeflow_metadata_config, preserving_proto_field_name=True),
        '--node_id',
        component.id,
        # TODO(b/182220464): write IR to pipeline_root and let
        # container_entrypoint.py read it back to avoid future issue that IR
        # exeeds the flag size limit.
        '--tfx_ir',
        json_format.MessageToJson(tfx_ir),
    ]

    for param in runtime_parameters:
      arguments.append('--runtime_parameter')
      arguments.append(_encode_runtime_parameter(param))

    self.container_op = dsl.ContainerOp(
        name=component.id,
        command=_COMMAND,
        image=tfx_image,
        arguments=arguments,
        output_artifact_paths={
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json',
        },
    )

    logging.info('Adding upstream dependencies for component %s',
                 self.container_op.name)
    for op in depends_on:
      logging.info('   ->  Component: %s', op.name)
      self.container_op.after(op)

    # TODO(b/140172100): Document the use of additional_pipeline_args.
    if _WORKFLOW_ID_KEY in pipeline.additional_pipeline_args:
      # Allow overriding pipeline's run_id externally, primarily for testing.
      self.container_op.container.add_env_variable(
          k8s_client.V1EnvVar(
              name=_WORKFLOW_ID_KEY,
              value=pipeline.additional_pipeline_args[_WORKFLOW_ID_KEY]))
    else:
      # Add the Argo workflow ID to the container's environment variable so it
      # can be used to uniquely place pipeline outputs under the pipeline_root.
      field_path = "metadata.labels['workflows.argoproj.io/workflow']"
      self.container_op.container.add_env_variable(
          k8s_client.V1EnvVar(
              name=_WORKFLOW_ID_KEY,
              value_from=k8s_client.V1EnvVarSource(
                  field_ref=k8s_client.V1ObjectFieldSelector(
                      field_path=field_path))))

    if pod_labels_to_attach:
      for k, v in pod_labels_to_attach.items():
        self.container_op.add_pod_label(k, v)
