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
"""TFX runner for Kubeflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Callable, Dict, List, Optional, Text, Type

from kfp import compiler
from kfp import dsl
from kfp import gcp
from kubernetes import client as k8s_client

from tfx import version
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.kubeflow import base_component
from tfx.orchestration.kubeflow import utils
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.orchestration.launcher import kubernetes_component_launcher
from tfx.utils import json_utils
from tfx.utils import telemetry_utils

# OpFunc represents the type of a function that takes as input a
# dsl.ContainerOp and returns the same object. Common operations such as adding
# k8s secrets, mounting volumes, specifying the use of TPUs and so on can be
# specified as an OpFunc.
# See example usage here:
# https://github.com/kubeflow/pipelines/blob/master/sdk/python/kfp/gcp.py
OpFunc = Callable[[dsl.ContainerOp], dsl.ContainerOp]

# Default secret name for GCP credentials. This secret is installed as part of
# a typical Kubeflow installation when the component is GKE.
_KUBEFLOW_GCP_SECRET_NAME = 'user-gcp-sa'

# Default TFX container image to use in KubeflowDagRunner.
_KUBEFLOW_TFX_IMAGE = 'tensorflow/tfx:%s' % (version.__version__)


def _mount_config_map_op(config_map_name: Text) -> OpFunc:
  """Mounts all key-value pairs found in the named Kubernetes ConfigMap.

  All key-value pairs in the ConfigMap are mounted as environment variables.

  Args:
    config_map_name: The name of the ConfigMap resource.

  Returns:
    An OpFunc for mounting the ConfigMap.
  """

  def mount_config_map(container_op: dsl.ContainerOp):
    config_map_ref = k8s_client.V1ConfigMapEnvSource(
        name=config_map_name, optional=True)
    container_op.container.add_env_from(
        k8s_client.V1EnvFromSource(config_map_ref=config_map_ref))

  return mount_config_map


def _mount_secret_op(secret_name: Text) -> OpFunc:
  """Mounts all key-value pairs found in the named Kubernetes Secret.

  All key-value pairs in the Secret are mounted as environment variables.

  Args:
    secret_name: The name of the Secret resource.

  Returns:
    An OpFunc for mounting the Secret.
  """

  def mount_secret(container_op: dsl.ContainerOp):
    secret_ref = k8s_client.V1ConfigMapEnvSource(
        name=secret_name, optional=True)

    container_op.container.add_env_from(
        k8s_client.V1EnvFromSource(secret_ref=secret_ref))

  return mount_secret


def get_default_pipeline_operator_funcs(
    use_gcp_sa: bool = False) -> List[OpFunc]:
  """Returns a default list of pipeline operator functions.

  Args:
    use_gcp_sa: If true, mount a GCP service account secret to each pod, with
      the name _KUBEFLOW_GCP_SECRET_NAME.

  Returns:
    A list of functions with type OpFunc.
  """
  # Enables authentication for GCP services if needed.
  gcp_secret_op = gcp.use_gcp_secret(_KUBEFLOW_GCP_SECRET_NAME)

  # Mounts configmap containing Metadata gRPC server configuration.
  mount_config_map_op = _mount_config_map_op('metadata-grpc-configmap')
  if use_gcp_sa:
    return [gcp_secret_op, mount_config_map_op]
  else:
    return [mount_config_map_op]


def get_default_kubeflow_metadata_config(
) -> kubeflow_pb2.KubeflowMetadataConfig:
  """Returns the default metadata connection config for Kubeflow.

  Returns:
    A config proto that will be serialized as JSON and passed to the running
    container so the TFX component driver is able to communicate with MLMD in
    a Kubeflow cluster.
  """
  # The default metadata configuration for a Kubeflow Pipelines cluster is
  # codified as a Kubernetes ConfigMap
  # https://github.com/kubeflow/pipelines/blob/master/manifests/kustomize/base/metadata/metadata-grpc-configmap.yaml

  config = kubeflow_pb2.KubeflowMetadataConfig()
  # The environment variable to use to obtain the Metadata gRPC service host in
  # the cluster that is backing Kubeflow Metadata. Note that the key in the
  # config map and therefore environment variable used, are lower-cased.
  config.grpc_config.grpc_service_host.environment_variable = 'METADATA_GRPC_SERVICE_HOST'
  # The environment variable to use to obtain the Metadata grpc service port in
  # the cluster that is backing Kubeflow Metadata.
  config.grpc_config.grpc_service_port.environment_variable = 'METADATA_GRPC_SERVICE_PORT'

  return config


def get_default_pod_labels() -> Dict[Text, Text]:
  """Returns the default pod label dict for Kubeflow."""
  # KFP default transformers add pod env:
  # https://github.com/kubeflow/pipelines/blob/0.1.32/sdk/python/kfp/compiler/_default_transformers.py
  result = {
      'add-pod-env': 'true',
      telemetry_utils.LABEL_KFP_SDK_ENV: 'tfx'
  }
  return result


class KubeflowDagRunnerConfig(pipeline_config.PipelineConfig):
  """Runtime configuration parameters specific to execution on Kubeflow."""

  def __init__(
      self,
      pipeline_operator_funcs: Optional[List[OpFunc]] = None,
      tfx_image: Optional[Text] = None,
      kubeflow_metadata_config: Optional[
          kubeflow_pb2.KubeflowMetadataConfig] = None,
      # TODO(b/143883035): Figure out the best practice to put the
      # SUPPORTED_LAUNCHER_CLASSES
      supported_launcher_classes: List[Type[
          base_component_launcher.BaseComponentLauncher]] = None,
      **kwargs):
    """Creates a KubeflowDagRunnerConfig object.

    The user can use pipeline_operator_funcs to apply modifications to
    ContainerOps used in the pipeline. For example, to ensure the pipeline
    steps mount a GCP secret, and a Persistent Volume, one can create config
    object like so:

      from kfp import gcp, onprem
      mount_secret_op = gcp.use_secret('my-secret-name)
      mount_volume_op = onprem.mount_pvc(
        "my-persistent-volume-claim",
        "my-volume-name",
        "/mnt/volume-mount-path")

      config = KubeflowDagRunnerConfig(
        pipeline_operator_funcs=[mount_secret_op, mount_volume_op]
      )

    Args:
      pipeline_operator_funcs: A list of ContainerOp modifying functions that
        will be applied to every container step in the pipeline.
      tfx_image: The TFX container image to use in the pipeline.
      kubeflow_metadata_config: Runtime configuration to use to connect to
        Kubeflow metadata.
      supported_launcher_classes: A list of component launcher classes that are
        supported by the current pipeline. List sequence determines the order in
        which launchers are chosen for each component being run.
      **kwargs: keyword args for PipelineConfig.
    """
    supported_launcher_classes = supported_launcher_classes or [
        in_process_component_launcher.InProcessComponentLauncher,
        kubernetes_component_launcher.KubernetesComponentLauncher,
    ]
    super(KubeflowDagRunnerConfig, self).__init__(
        supported_launcher_classes=supported_launcher_classes, **kwargs)
    self.pipeline_operator_funcs = (
        pipeline_operator_funcs or get_default_pipeline_operator_funcs())
    self.tfx_image = tfx_image or _KUBEFLOW_TFX_IMAGE
    self.kubeflow_metadata_config = (
        kubeflow_metadata_config or get_default_kubeflow_metadata_config())


class KubeflowDagRunner(tfx_runner.TfxRunner):
  """Kubeflow Pipelines runner.

  Constructs a pipeline definition YAML file based on the TFX logical pipeline.
  """

  def __init__(
      self,
      output_dir: Optional[Text] = None,
      output_filename: Optional[Text] = None,
      config: Optional[KubeflowDagRunnerConfig] = None,
      pod_labels_to_attach: Optional[Dict[Text, Text]] = None
  ):
    """Initializes KubeflowDagRunner for compiling a Kubeflow Pipeline.

    Args:
      output_dir: An optional output directory into which to output the pipeline
        definition files. Defaults to the current working directory.
      output_filename: An optional output file name for the pipeline definition
        file. Defaults to pipeline_name.tar.gz when compiling a TFX pipeline.
        Currently supports .tar.gz, .tgz, .zip, .yaml, .yml formats. See
        https://github.com/kubeflow/pipelines/blob/181de66cf9fa87bcd0fe9291926790c400140783/sdk/python/kfp/compiler/compiler.py#L851
          for format restriction.
      config: An optional KubeflowDagRunnerConfig object to specify runtime
        configuration when running the pipeline under Kubeflow.
      pod_labels_to_attach: Optional set of pod labels to attach to GKE pod
        spinned up for this pipeline. Default to the 3 labels:
        1. add-pod-env: true,
        2. pipeline SDK type,
        3. pipeline unique ID,
        where 2 and 3 are instrumentation of usage tracking.
    """
    if config and not isinstance(config, KubeflowDagRunnerConfig):
      raise TypeError('config must be type of KubeflowDagRunnerConfig.')
    super(KubeflowDagRunner, self).__init__(config or KubeflowDagRunnerConfig())
    self._output_dir = output_dir or os.getcwd()
    self._output_filename = output_filename
    self._compiler = compiler.Compiler()
    self._params = []  # List of dsl.PipelineParam used in this pipeline.
    self._deduped_parameter_names = set()  # Set of unique param names used.
    if pod_labels_to_attach is None:
      self._pod_labels_to_attach = get_default_pod_labels()
    else:
      self._pod_labels_to_attach = pod_labels_to_attach

  def _parse_parameter_from_component(
      self, component: base_component.BaseComponent) -> None:
    """Extract embedded RuntimeParameter placeholders from a component.

    Extract embedded RuntimeParameter placeholders from a component, then append
    the corresponding dsl.PipelineParam to KubeflowDagRunner.

    Args:
      component: a TFX component.
    """

    serialized_component = json_utils.dumps(component)
    placeholders = re.findall(data_types.RUNTIME_PARAMETER_PATTERN,
                              serialized_component)
    for placeholder in placeholders:
      placeholder = placeholder.replace('\\', '')  # Clean escapes.
      placeholder = utils.fix_brackets(placeholder)  # Fix brackets if needed.
      parameter = json_utils.loads(placeholder)
      # Escape pipeline root because it will be added later.
      if parameter.name == tfx_pipeline.ROOT_PARAMETER.name:
        continue
      if parameter.name not in self._deduped_parameter_names:
        self._deduped_parameter_names.add(parameter.name)
        dsl_parameter = dsl.PipelineParam(
            name=parameter.name, value=parameter.default)
        self._params.append(dsl_parameter)

  def _parse_parameter_from_pipeline(self,
                                     pipeline: tfx_pipeline.Pipeline) -> None:
    """Extract all the RuntimeParameter placeholders from the pipeline."""

    for component in pipeline.components:
      self._parse_parameter_from_component(component)

  def _construct_pipeline_graph(self, pipeline: tfx_pipeline.Pipeline,
                                pipeline_root: dsl.PipelineParam):
    """Constructs a Kubeflow Pipeline graph.

    Args:
      pipeline: The logical TFX pipeline to base the construction on.
      pipeline_root: dsl.PipelineParam representing the pipeline root.
    """
    component_to_kfp_op = {}

    # Assumption: There is a partial ordering of components in the list, i.e.,
    # if component A depends on component B and C, then A appears after B and C
    # in the list.
    for component in pipeline.components:
      # Keep track of the set of upstream dsl.ContainerOps for this component.
      depends_on = set()

      for upstream_component in component.upstream_nodes:
        depends_on.add(component_to_kfp_op[upstream_component])

      (component_launcher_class,
       component_config) = config_utils.find_component_launch_info(
           self._config, component)

      kfp_component = base_component.BaseComponent(
          component=component,
          component_launcher_class=component_launcher_class,
          depends_on=depends_on,
          pipeline=pipeline,
          pipeline_name=pipeline.pipeline_info.pipeline_name,
          pipeline_root=pipeline_root,
          tfx_image=self._config.tfx_image,
          kubeflow_metadata_config=self._config.kubeflow_metadata_config,
          component_config=component_config,
          pod_labels_to_attach=self._pod_labels_to_attach)

      for operator in self._config.pipeline_operator_funcs:
        kfp_component.container_op.apply(operator)

      component_to_kfp_op[component] = kfp_component.container_op

  def run(self, pipeline: tfx_pipeline.Pipeline):
    """Compiles and outputs a Kubeflow Pipeline YAML definition file.

    Args:
      pipeline: The logical TFX pipeline to use when building the Kubeflow
        pipeline.
    """
    pipeline_root = tfx_pipeline.ROOT_PARAMETER
    # KFP DSL representation of pipeline root parameter.
    dsl_pipeline_root = dsl.PipelineParam(
        name=pipeline_root.name, value=pipeline.pipeline_info.pipeline_root)
    self._params.append(dsl_pipeline_root)

    def _construct_pipeline():
      """Constructs a Kubeflow pipeline.

      Creates Kubeflow ContainerOps for each TFX component encountered in the
      logical pipeline definition.
      """
      self._construct_pipeline_graph(pipeline, dsl_pipeline_root)

    # Need to run this first to get self._params populated. Then KFP compiler
    # can correctly match default value with PipelineParam.
    self._parse_parameter_from_pipeline(pipeline)

    file_name = self._output_filename or pipeline.pipeline_info.pipeline_name + '.tar.gz'
    # Create workflow spec and write out to package.
    self._compiler._create_and_write_workflow(  # pylint: disable=protected-access
        pipeline_func=_construct_pipeline,
        pipeline_name=pipeline.pipeline_info.pipeline_name,
        params_list=self._params,
        package_path=os.path.join(self._output_dir, file_name))
