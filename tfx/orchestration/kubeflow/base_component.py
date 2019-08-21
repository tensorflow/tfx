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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from kfp import dsl
from kubernetes import client as k8s_client
import tensorflow as tf
from typing import Optional, Set, Text, Type

from tfx.components.base import base_component as tfx_base_component
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.launcher import base_component_launcher
from tfx.types import artifact_utils
from tfx.types import component_spec
from tfx.utils import json_utils
from google.protobuf import json_format

_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py'
]

_WORKFLOW_ID_KEY = 'WORKFLOW_ID'


def _prepare_artifact_dict(wrapper: component_spec._PropertyDictWrapper):
  return dict((k, v.get()) for k, v in wrapper.get_all().items())


class BaseComponent(object):
  """Base component for all Kubeflow pipelines TFX components.

  Returns a wrapper around a KFP DSL ContainerOp class, and adds named output
  attributes that match the output names for the corresponding native TFX
  components.
  """

  def __init__(
      self,
      component: tfx_base_component.BaseComponent,
      component_launcher_class: Type[
          base_component_launcher.BaseComponentLauncher],
      depends_on: Set[dsl.ContainerOp],
      pipeline: tfx_pipeline.Pipeline,
      tfx_image: Text,
      kubeflow_metadata_config: Optional[kubeflow_pb2.KubeflowMetadataConfig],
  ):
    """Creates a new Kubeflow-based component.

    This class essentially wraps a dsl.ContainerOp construct in Kubeflow
    Pipelines.

    Args:
      component: The logical TFX component to wrap.
      component_launcher_class: the class of the launcher to launch the
        component.
      depends_on: The set of upstream KFP ContainerOp components that this
        component will depend on.
      pipeline: The logical TFX pipeline to which this component belongs.
      tfx_image: The container image to use for this component.
      kubeflow_metadata_config: Configuration settings for
        connecting to the MLMD store in a Kubeflow cluster.
    """
    driver_class_path = '.'.join(
        [component.driver_class.__module__, component.driver_class.__name__])
    executor_spec = json_utils.dumps(component.executor_spec)
    component_launcher_class_path = '.'.join([
        component_launcher_class.__module__, component_launcher_class.__name__
    ])

    # pyformat: disable
    arguments = [
        '--pipeline_name', pipeline.pipeline_info.pipeline_name,
        '--pipeline_root', pipeline.pipeline_info.pipeline_root,
        '--kubeflow_metadata_config',
        json_format.MessageToJson(kubeflow_metadata_config),
        '--additional_pipeline_args',
        json.dumps(pipeline.additional_pipeline_args),
        '--component_id',
        component.component_id,
        '--component_type',
        component.component_type,
        '--driver_class_path',
        driver_class_path,
        '--executor_spec',
        executor_spec,
        '--component_launcher_class_path',
        component_launcher_class_path,
        '--inputs',
        artifact_utils.jsonify_artifact_dict(
            _prepare_artifact_dict(component.inputs)),
        '--outputs',
        artifact_utils.jsonify_artifact_dict(
            _prepare_artifact_dict(component.outputs)),
        '--exec_properties', json.dumps(component.exec_properties),
    ]
    # pyformat: enable

    if pipeline.enable_cache:
      arguments.append('--enable_cache')

    self.container_op = dsl.ContainerOp(
        name=component.component_id.replace('.', '_'),
        command=_COMMAND,
        image=tfx_image,
        arguments=arguments,
    )

    tf.logging.info('Adding upstream dependencies for component {}'.format(
        self.container_op.name))
    for op in depends_on:
      tf.logging.info('   ->  Component: {}'.format(op.name))
      self.container_op.after(op)

    # TODO(b/140172100): Document the use of additional_pipeline_args.
    if _WORKFLOW_ID_KEY in pipeline.additional_pipeline_args:
      # Allow overriding pipeline's run_id externally, primarily for testing.
      self.container_op.add_env_variable(
          k8s_client.V1EnvVar(
              name=_WORKFLOW_ID_KEY,
              value=pipeline.additional_pipeline_args[_WORKFLOW_ID_KEY]))
    else:
      # Add the Argo workflow ID to the container's environment variable so it
      # can be used to uniquely place pipeline outputs under the pipeline_root.
      field_path = "metadata.labels['workflows.argoproj.io/workflow']"
      self.container_op.add_env_variable(
          k8s_client.V1EnvVar(
              name=_WORKFLOW_ID_KEY,
              value_from=k8s_client.V1EnvVarSource(
                  field_ref=k8s_client.V1ObjectFieldSelector(
                      field_path=field_path))))
