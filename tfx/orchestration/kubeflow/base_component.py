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
from kfp import gcp
from kubernetes import client as k8s_client
from tfx.utils import types


class ExecutionProperties(object):
  """Holds global execution properties that apply to all component."""
  exec_properties = {}

  def __new__(cls,
              output_dir,
              log_root,
              beam_pipeline_args = None):
    cls.exec_properties = {
        'output_dir': output_dir,
        'log_root': log_root,
    }
    if beam_pipeline_args is not None:
      cls.exec_properties['beam_pipeline_args'] = beam_pipeline_args

    return super(ExecutionProperties, cls).__new__(cls)


# TODO(ajaygopinathan): Update with final image location.
_KUBEFLOW_TFX_IMAGE = 'gcr.io/cloud-ml-pipelines-test/tfx-kubeflow'
_COMMAND = [
    'python', '/tfx-src/tfx/orchestration/kubeflow/container_entrypoint.py'
]


class BaseComponent(object):
  """Base component for all Kubeflow pipelines TFX components.

  Returns a wrapper around a KFP DSL ContainerOp class, and adds named output
  attributes that match the output names for the corresponding native TFX
  components.
  """

  def __new__(cls, component_name, input_dict,
              output_dict,
              exec_properties):
    """Creates a new component.

    Args:
      component_name: TFX component name.
      input_dict: Dictionary of input names to TFX types, or
        kfp.dsl.PipelineParam representing input parameters.
      output_dict: Dictionary of output names to List of TFX types.
      exec_properties: Execution properties.

    Returns:
      Newly constructed TFX Kubeflow component instance.
    """
    outputs = output_dict.keys()
    file_outputs = {
        output: '/output/ml_metadata/{}'.format(output) for output in outputs
    }

    for k, v in ExecutionProperties.exec_properties.items():
      exec_properties[k] = v

    arguments = [
        '--exec_properties',
        json.dumps(exec_properties),
        '--outputs',
        types.jsonify_tfx_type_dict(output_dict),
        component_name,
    ]

    for k, v in input_dict.items():
      if isinstance(v, float) or isinstance(v, int):
        v = str(v)
      arguments.append('--{}'.format(k))
      arguments.append(v)

    container_op = dsl.ContainerOp(
        name=component_name,
        command=_COMMAND,
        image=_KUBEFLOW_TFX_IMAGE,
        arguments=arguments,
        file_outputs=file_outputs,
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))  # Adds GCP authentication.

    # Add the Argo workflow ID to the container's environment variable so it
    # can be used to uniquely place pipeline outputs under the pipeline_root.
    field_path = "metadata.labels['workflows.argoproj.io/workflow']"
    container_op.add_env_variable(
        k8s_client.V1EnvVar(
            name='WORKFLOW_ID',
            value_from=k8s_client.V1EnvVarSource(
                field_ref=k8s_client.V1ObjectFieldSelector(
                    field_path=field_path))))

    named_outputs = {output: container_op.outputs[output] for output in outputs}

    # This allows user code to refer to the ContainerOp 'op' output named 'x'
    # as op.outputs.x
    component_outputs = type('Output', (), named_outputs)

    return type(component_name, (BaseComponent,), {
        'container_op': container_op,
        'outputs': component_outputs
    })
