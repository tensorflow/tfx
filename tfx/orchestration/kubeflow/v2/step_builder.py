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
"""Builder for Kubeflow pipelines StepSpec proto."""

from typing import Any, Dict, List, Optional, Text, Tuple

from tfx import components
from tfx.components.evaluator import constants
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import compiler_utils
from tfx.orchestration.kubeflow.v2.proto import pipeline_pb2
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2

_EXECUTOR_LABEL_PATTERN = '{}_executor'

# Task name suffix used for the ModelBlessing resolver part of
# latest blessed model resolver.
_MODEL_BLESSING_RESOLVER_SUFFIX = '-model-blessing-resolver'
# Task name suffix used for Model resolver part of latest blessed model
# resolver.
_MODEL_RESOLVER_SUFFIX = '-model-resolver'

# Input key used in model resolver spec.
_MODEL_RESOLVER_INPUT_KEY = 'input'

# Shorthands for various specs in Kubeflow IR proto.
ResolverSpec = pipeline_pb2.PipelineDeploymentConfig.ResolverSpec
ImporterSpec = pipeline_pb2.PipelineDeploymentConfig.ImporterSpec
ContainerSpec = pipeline_pb2.PipelineDeploymentConfig.PipelineContainerSpec

_DRIVER_COMMANDS = (
    'python', '-m',
    'tfx.orchestration.kubeflow.v2.file_based_example_gen.driver')


def _resolve_command_line(
    container_spec: executor_specs.TemplatedExecutorContainerSpec,
    exec_properties: Dict[Text, Any],
) -> List[Text]:
  """Resolves placeholders in the command line of a container.

  Args:
    container_spec: Container structure to resolve
    exec_properties: The map of component's execution properties

  Returns:
    Resolved command line.

  Raises:
    TypeError: On unsupported type of command-line arguments, or when the
      resolved argument is not a string.
  """

  def expand_command_line_arg(
      cmd_arg: placeholders.CommandlineArgumentType) -> Text:
    """Resolves a single argument."""
    if isinstance(cmd_arg, str):
      return cmd_arg
    elif isinstance(cmd_arg, placeholders.InputValuePlaceholder):
      if cmd_arg.input_name in exec_properties:
        return "{{$.inputs.parameters['%s']}}" % cmd_arg.input_name
      else:
        return "{{$.inputs.artifacts['%s'].value}}" % cmd_arg.input_name
    elif isinstance(cmd_arg, placeholders.InputUriPlaceholder):
      return "{{$.inputs.artifacts['%s'].uri}}" % cmd_arg.input_name
    elif isinstance(cmd_arg, placeholders.OutputUriPlaceholder):
      return "{{$.outputs.artifacts['%s'].uri}}" % cmd_arg.output_name
    elif isinstance(cmd_arg, placeholders.ConcatPlaceholder):
      resolved_items = [expand_command_line_arg(item) for item in cmd_arg.items]
      for item in resolved_items:
        if not isinstance(item, (str, Text)):
          raise TypeError('Expanded item "{}" has incorrect type "{}"'.format(
              item, type(item)))
      return ''.join(resolved_items)
    else:
      raise TypeError('Unsupported type of command-line arguments: "{}".'
                      ' Supported types are {}.'.format(
                          type(cmd_arg),
                          str(executor_specs.CommandlineArgumentType)))

  resolved_command_line = []
  for cmd_arg in (container_spec.command or []):
    resolved_cmd_arg = expand_command_line_arg(cmd_arg)
    if not isinstance(resolved_cmd_arg, (str, Text)):
      raise TypeError(
          'Resolved argument "{}" (type="{}") is not a string.'.format(
              resolved_cmd_arg, type(resolved_cmd_arg)))
    resolved_command_line.append(resolved_cmd_arg)

  return resolved_command_line


class StepBuilder(object):
  """Kubeflow pipelines task builder.

  Constructs a pipeline task spec based on the TFX node information. Meanwhile,
  augments the deployment config associated with the node.
  """

  def __init__(self,
               node: base_node.BaseNode,
               deployment_config: pipeline_pb2.PipelineDeploymentConfig,
               image: Optional[Text] = None,
               image_cmds: Optional[List[Text]] = None,
               beam_pipeline_args: Optional[List[Text]] = None,
               enable_cache: bool = False,
               pipeline_info: Optional[data_types.PipelineInfo] = None,
               channel_redirect_map: Optional[Dict[Tuple[Text, Text],
                                                   Text]] = None):
    """Creates a StepBuilder object.

    A StepBuilder takes in a TFX node object (usually it's a component/resolver/
    importer), together with other configuration (e.g., TFX image, beam args).
    Then, step_builder.build() outputs the StepSpec pb object.

    Args:
      node: A TFX node. The logical unit of a step. Note, currently for resolver
        node we only support two types of resolver
        policies, including: 1) latest blessed model, and 2) latest model
          artifact.
      deployment_config: The deployment config in Kubeflow IR to be populated.
      image: TFX image used in the underlying container spec. Required if node
        is a TFX component.
      image_cmds: Optional. If not specified the default `ENTRYPOINT` defined
        in the docker image will be used. Note: the commands here refers to the
          K8S container command, which maps to Docker entrypoint field. If one
          supplies command but no args are provided for the container, the
          container will be invoked with the provided command, ignoring the
          `ENTRYPOINT` and `CMD` defined in the Dockerfile. One can find more
          details regarding the difference between K8S and Docker conventions at
        https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#notes
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      enable_cache: If true, enables cache lookup for this pipeline step.
        Defaults to False.
      pipeline_info: Optionally, the pipeline info associated with current
        pipeline. The pipeline context is required if the current node is a
        resolver. Defaults to None.
      channel_redirect_map: Map from (producer component id, output key) to (new
        producer component id, output key). This is needed for cases where one
        DSL node is splitted into multiple tasks in pipeline API proto. For
        example, latest blessed model resolver.

    Raises:
      ValueError: On the following two cases:
        1. The node being built is an instance of BaseComponent but image was
           not provided.
        2. The node being built is a Resolver but the associated pipeline
           info was not provided.
    """
    self._name = node.id
    self._node = node
    self._deployment_config = deployment_config
    self._inputs = node.inputs.get_all()  # type: Dict[Text, channel.Channel]
    self._outputs = node.outputs.get_all()  # type: Dict[Text, channel.Channel]
    self._enable_cache = enable_cache
    if channel_redirect_map is None:
      self._channel_redirect_map = {}
    else:
      self._channel_redirect_map = channel_redirect_map

    self._exec_properties = node.exec_properties

    if isinstance(self._node, base_component.BaseComponent) and not image:
      raise ValueError('TFX image is required for component of type %s' %
                       type(self._node))
    if isinstance(self._node, resolver.Resolver) and not pipeline_info:
      raise ValueError('pipeline_info is needed for resolver node.')

    self._tfx_image = image
    self._image_cmds = image_cmds
    self._beam_pipeline_args = beam_pipeline_args or []
    self._pipeline_info = pipeline_info

  def build(self) -> List[pipeline_pb2.PipelineTaskSpec]:
    """Builds a pipeline StepSpec given the node information.

    Returns:
      A list of PipelineTaskSpec messages corresponding to the node. For most of
      the cases, the list contains a single element. The only exception is when
      compiling latest blessed model resolver. One DSL node will be
      split to two resolver specs to reflect the two-phased query execution.
    Raises:
      NotImplementedError: When the node being built is an InfraValidator.
    """
    task_spec = pipeline_pb2.PipelineTaskSpec()
    task_spec.task_info.CopyFrom(pipeline_pb2.PipelineTaskInfo(name=self._name))
    executor_label = _EXECUTOR_LABEL_PATTERN.format(self._name)
    task_spec.executor_label = executor_label
    executor = pipeline_pb2.PipelineDeploymentConfig.ExecutorSpec()

    # 1. Resolver tasks won't have input artifacts in the API proto. First we
    #    specialcase two resolver types we support.
    if isinstance(self._node, resolver.Resolver):
      return self._build_resolver_spec()

    # 2. Build the node spec.
    # TODO(b/157641727): Tests comparing dictionaries are brittle when comparing
    # lists as ordering matters.
    dependency_ids = [node.id for node in self._node.upstream_nodes]
    # Specify the inputs of the task.
    for name, input_channel in self._inputs.items():
      # If the redirecting map is provided (usually for latest blessed model
      # resolver, we'll need to redirect accordingly. Also, the upstream node
      # list will be updated and replaced by the new producer id.
      producer_id = input_channel.producer_component_id
      output_key = input_channel.output_key
      for k, v in self._channel_redirect_map.items():
        if k[0] == producer_id and producer_id in dependency_ids:
          dependency_ids.remove(producer_id)
          dependency_ids.append(v[0])
      producer_id = self._channel_redirect_map.get((producer_id, output_key),
                                                   (producer_id, output_key))[0]
      output_key = self._channel_redirect_map.get((producer_id, output_key),
                                                  (producer_id, output_key))[1]

      input_artifact_spec = pipeline_pb2.TaskInputsSpec.InputArtifactSpec(
          producer_task=producer_id, output_artifact_key=output_key)
      task_spec.inputs.artifacts[name].CopyFrom(input_artifact_spec)

    # Specify the outputs of the task.
    for name, output_channel in self._outputs.items():
      # Currently, we're working under the assumption that for tasks
      # (those generated by BaseComponent), each channel contains a single
      # artifact.
      output_artifact_spec = compiler_utils.build_output_artifact_spec(
          output_channel)
      task_spec.outputs.artifacts[name].CopyFrom(output_artifact_spec)

    # Specify the input parameters of the task.
    for k, v in compiler_utils.build_input_parameter_spec(
        self._exec_properties).items():
      task_spec.inputs.parameters[k].CopyFrom(v)

    # 3. Build the executor body for other common tasks.
    if isinstance(self._node, importer.Importer):
      executor.importer.CopyFrom(self._build_importer_spec())
    elif isinstance(self._node, components.FileBasedExampleGen):
      executor.container.CopyFrom(self._build_file_based_example_gen_spec())
    elif isinstance(self._node, (components.InfraValidator)):
      raise NotImplementedError(
          'The componet type "{}" is not supported'.format(type(self._node)))
    else:
      executor.container.CopyFrom(self._build_container_spec())

    dependency_ids = sorted(dependency_ids)
    for dependency in dependency_ids:
      task_spec.dependent_tasks.append(dependency)

    task_spec.caching_options.CopyFrom(
        pipeline_pb2.PipelineTaskSpec.CachingOptions(
            enable_cache=self._enable_cache))

    # 4. Attach the built executor spec to the deployment config.
    self._deployment_config.executors[executor_label].CopyFrom(executor)

    return [task_spec]

  def _build_container_spec(self) -> ContainerSpec:
    """Builds the container spec for a component.

    Returns:
      The PipelineContainerSpec represents the container execution of the
      component.

    Raises:
      NotImplementedError: When the executor class is neither ExecutorClassSpec
      nor TemplatedExecutorContainerSpec.
    """
    assert isinstance(self._node, base_component.BaseComponent)
    if isinstance(self._node.executor_spec,
                  executor_specs.TemplatedExecutorContainerSpec):
      container_spec = self._node.executor_spec
      result = ContainerSpec(
          image=container_spec.image,
          command=_resolve_command_line(
              container_spec=container_spec,
              exec_properties=self._node.exec_properties,
          ),
      )
      return result

    # The container entrypoint format below assumes ExecutorClassSpec.
    if not isinstance(self._node.executor_spec,
                      executor_spec.ExecutorClassSpec):
      raise NotImplementedError(
          'Executor spec: % is not supported in Kubeflow V2 yet.'
          'Currently only ExecutorClassSpec is supported.')

    result = ContainerSpec()
    result.image = self._tfx_image
    if self._image_cmds:
      for cmd in self._image_cmds:
        result.command.append(cmd)
    executor_path = '%s.%s' % (
        self._node.executor_spec.executor_class.__module__,
        self._node.executor_spec.executor_class.__name__)
    # Resolve container arguments.
    result.args.append('--executor_class_path')
    result.args.append(executor_path)
    result.args.append('--json_serialized_invocation_args')
    result.args.append('{{$}}')
    result.args.extend(self._beam_pipeline_args)

    return result

  def _build_file_based_example_gen_spec(self) -> ContainerSpec:
    """Builds FileBasedExampleGen into a PipelineContainerSpec.

    Returns:
      The PipelineContainerSpec represents the container execution of the
      component, which should includes both the driver execution, and the
      executor execution.

    Raises:
      ValueError: When the node is a FileBasedExampleGen but tfx image was not
        specified.
    """
    assert isinstance(self._node, components.FileBasedExampleGen)
    if not self._tfx_image:
      raise ValueError('TFX image is required for FileBasedExampleGen.')

    # 1. Build the driver container execution by setting the pre_cache_check
    # hook.
    result = ContainerSpec()
    driver_hook = ContainerSpec.Lifecycle(
        pre_cache_check=ContainerSpec.Lifecycle.Exec(
            command=_DRIVER_COMMANDS,
            args=[
                '--json_serialized_invocation_args',
                '{{$}}',
            ]))
    driver_hook.pre_cache_check.args.extend(self._beam_pipeline_args)
    result.lifecycle.CopyFrom(driver_hook)

    # 2. Build the executor container execution in the same way as a regular
    # component.
    result.image = self._tfx_image
    if self._image_cmds:
      for cmd in self._image_cmds:
        result.command.append(cmd)
    executor_path = '%s.%s' % (
        self._node.executor_spec.executor_class.__module__,
        self._node.executor_spec.executor_class.__name__)
    # Resolve container arguments.
    result.args.append('--executor_class_path')
    result.args.append(executor_path)
    result.args.append('--json_serialized_invocation_args')
    result.args.append('{{$}}')
    result.args.extend(self._beam_pipeline_args)
    return result

  def _build_importer_spec(self) -> ImporterSpec:
    """Builds ImporterSpec."""
    assert isinstance(self._node, importer.Importer)
    result = ImporterSpec(
        properties=compiler_utils.convert_from_tfx_properties(
            self._exec_properties[importer.PROPERTIES_KEY]),
        custom_properties=compiler_utils.convert_from_tfx_properties(
            self._exec_properties[importer.CUSTOM_PROPERTIES_KEY]))
    result.reimport = bool(self._exec_properties[importer.REIMPORT_OPTION_KEY])
    result.artifact_uri.CopyFrom(
        compiler_utils.value_converter(
            self._exec_properties[importer.SOURCE_URI_KEY]))
    single_artifact = artifact_utils.get_single_instance(
        list(self._node.outputs[importer.IMPORT_RESULT_KEY].get()))
    result.type_schema.CopyFrom(
        pipeline_pb2.ArtifactTypeSchema(
            instance_schema=compiler_utils.get_artifact_schema(
                single_artifact)))

    return result

  def _build_latest_artifact_resolver(
      self) -> List[pipeline_pb2.PipelineTaskSpec]:
    """Builds a resolver spec for a latest artifact resolver.

    Returns:
      A list of two PipelineTaskSpecs. One represents the query for latest valid
      ModelBlessing artifact. Another one represents the query for latest
      blessed Model artifact.
    Raises:
      ValueError: when desired_num_of_artifacts != 1. 1 is the only supported
        value currently.
    """

    task_spec = pipeline_pb2.PipelineTaskSpec()
    task_spec.task_info.CopyFrom(pipeline_pb2.PipelineTaskInfo(name=self._name))
    executor_label = _EXECUTOR_LABEL_PATTERN.format(self._name)
    task_spec.executor_label = executor_label

    # Fetch the init kwargs for the resolver.
    resolver_config = self._exec_properties[resolver.RESOLVER_CONFIG]
    if (isinstance(resolver_config, dict) and
        resolver_config.get('desired_num_of_artifacts', 0) > 1):
      raise ValueError('Only desired_num_of_artifacts=1 is supported currently.'
                       ' Got {}'.format(
                           resolver_config.get('desired_num_of_artifacts')))

    # Specify the outputs of the task.
    for name, output_channel in self._outputs.items():
      # Currently, we're working under the assumption that for tasks
      # (those generated by BaseComponent), each channel contains a single
      # artifact.
      output_artifact_spec = compiler_utils.build_output_artifact_spec(
          output_channel)
      task_spec.outputs.artifacts[name].CopyFrom(output_artifact_spec)

    # Specify the input parameters of the task.
    for k, v in compiler_utils.build_input_parameter_spec(
        self._exec_properties).items():
      task_spec.inputs.parameters[k].CopyFrom(v)

    artifact_queries = {}
    # Buid the artifact query for each channel in the input dict.
    for name, c in self._inputs.items():
      query_filter = ("artifact_type='{type}' and state={state}").format(
          type=compiler_utils.get_artifact_title(c.type),
          state=metadata_store_pb2.Artifact.State.Name(
              metadata_store_pb2.Artifact.LIVE))
      artifact_queries[name] = ResolverSpec.ArtifactQuerySpec(
          filter=query_filter)

    resolver_spec = ResolverSpec(output_artifact_queries=artifact_queries)
    executor = pipeline_pb2.PipelineDeploymentConfig.ExecutorSpec()
    executor.resolver.CopyFrom(resolver_spec)
    self._deployment_config.executors[executor_label].CopyFrom(executor)
    return [task_spec]

  def _build_resolver_for_latest_model_blessing(
      self, model_blessing_channel_key: str) -> pipeline_pb2.PipelineTaskSpec:
    """Builds the resolver spec for latest valid ModelBlessing artifact."""
    # 1. Build the task info.
    result = pipeline_pb2.PipelineTaskSpec()
    name = '{}{}'.format(self._name, _MODEL_BLESSING_RESOLVER_SUFFIX)
    result.task_info.CopyFrom(pipeline_pb2.PipelineTaskInfo(name=name))
    executor_label = _EXECUTOR_LABEL_PATTERN.format(name)
    result.executor_label = executor_label

    # 2. Specify the outputs of the task.
    result.outputs.artifacts[model_blessing_channel_key].CopyFrom(
        compiler_utils.build_output_artifact_spec(
            self._outputs[model_blessing_channel_key]))

    # 3. Build the resolver executor spec for latest valid ModelBlessing.
    executor = pipeline_pb2.PipelineDeploymentConfig.ExecutorSpec()
    artifact_queries = {}
    query_filter = ("artifact_type='{type}' and state={state}"
                    " and custom_properties['{key}']='{value}'").format(
                        type=compiler_utils.get_artifact_title(
                            standard_artifacts.ModelBlessing),
                        state=metadata_store_pb2.Artifact.State.Name(
                            metadata_store_pb2.Artifact.LIVE),
                        key=constants.ARTIFACT_PROPERTY_BLESSED_KEY,
                        value=constants.BLESSED_VALUE)
    artifact_queries[
        model_blessing_channel_key] = ResolverSpec.ArtifactQuerySpec(
            filter=query_filter)
    executor.resolver.CopyFrom(
        ResolverSpec(output_artifact_queries=artifact_queries))

    self._deployment_config.executors[executor_label].CopyFrom(executor)
    return result

  def _build_resolver_for_latest_blessed_model(
      self, model_channel_key: str, model_blessing_resolver_name: str,
      model_blessing_channel_key: str) -> pipeline_pb2.PipelineTaskSpec:
    """Builds the resolver spec for latest blessed Model artifact."""
    # 1. Build the task info.
    result = pipeline_pb2.PipelineTaskSpec()
    name = '{}{}'.format(self._name, _MODEL_RESOLVER_SUFFIX)
    result.task_info.CopyFrom(pipeline_pb2.PipelineTaskInfo(name=name))
    executor_label = _EXECUTOR_LABEL_PATTERN.format(name)
    result.executor_label = executor_label

    # 2. Specify the input of the task. The output from model_blessing_resolver
    # will be used as the input.
    input_artifact_spec = pipeline_pb2.TaskInputsSpec.InputArtifactSpec(
        producer_task=model_blessing_resolver_name,
        output_artifact_key=model_blessing_channel_key)
    result.inputs.artifacts[_MODEL_RESOLVER_INPUT_KEY].CopyFrom(
        input_artifact_spec)

    # 3. Specify the outputs of the task. model_resolver has one output for
    # the latest blessed model.
    result.outputs.artifacts[model_channel_key].CopyFrom(
        compiler_utils.build_output_artifact_spec(
            self._outputs[model_channel_key]))

    # 4. Build the resolver executor spec for latest blessed Model.
    executor = pipeline_pb2.PipelineDeploymentConfig.ExecutorSpec()
    artifact_queries = {}
    query_filter = (
        "artifact_type='{type}' and "
        "state={state} and name={{$.inputs.artifacts['{input_key}']"
        ".custom_properties['{property_key}']}}").format(
            type=compiler_utils.get_artifact_title(standard_artifacts.Model),
            state=metadata_store_pb2.Artifact.State.Name(
                metadata_store_pb2.Artifact.LIVE),
            input_key=_MODEL_RESOLVER_INPUT_KEY,
            property_key=constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY)
    artifact_queries[model_channel_key] = ResolverSpec.ArtifactQuerySpec(
        filter=query_filter)
    executor.resolver.CopyFrom(
        ResolverSpec(output_artifact_queries=artifact_queries))
    self._deployment_config.executors[executor_label].CopyFrom(executor)

    return result

  def _build_latest_blessed_model_resolver(
      self) -> List[pipeline_pb2.PipelineTaskSpec]:
    """Builds two resolver specs to resolve the latest blessed model."""
    # The two phased resolution logic will be mapped to the following:
    # 1. A ResolverSpec to get the latest ModelBlessing artifact under the
    #    current context, with a valid current_model_id custom property
    #    attached; and
    # 2. A ResolverSpec to get the latest Model artifact under the current
    #    context, where the unique name of the artifact is corresponding to
    #    the current_model_id field in step 1.
    #
    # Such conversion will generate two PipelineTaskSpec:
    # 1. A TaskSpec with the name '{component_id}-model-blessing-resolver',
    #    where component_id is the original node id from the DSL resolver node.
    #    This TaskSpec has no input artifact but one output artifacts,
    #    which is the latest valid ModelBlessing artifact it
    #    finds.
    # 2. A TaskSpec with the name '{component_id}-model-resolver'. This TaskSpec
    #    has one input artifact connected to the 'model_blessing' output of
    #    '{component_id}-model-blessing-resolver', representing the latest
    #    blessed model it finds in MLMD under the same context.
    assert len(self._inputs) == 2, 'Expecting 2 input Channels'

    model_channel_key, model_blessing_channel_key = None, None
    # Find the output key for ModelBlessing and Model, respectively
    for name, c in self._inputs.items():
      if issubclass(c.type, standard_artifacts.ModelBlessing):
        model_blessing_channel_key = name
      elif issubclass(c.type, standard_artifacts.Model):
        model_channel_key = name

    assert model_channel_key is not None, 'Expecting Model as input'
    assert model_blessing_channel_key is not None, ('Expecting ModelBlessing as'
                                                    ' input')

    model_blessing_resolver = self._build_resolver_for_latest_model_blessing(
        model_blessing_channel_key)

    model_resolver = self._build_resolver_for_latest_blessed_model(
        model_channel_key=model_channel_key,
        model_blessing_resolver_name=model_blessing_resolver.task_info.name,
        model_blessing_channel_key=model_blessing_channel_key)

    # 5. Modify the channel_redirect_map passed in.
    self._channel_redirect_map[(self._node.id, model_channel_key)] = (
        model_resolver.task_info.name, model_channel_key)
    self._channel_redirect_map[(self._node.id, model_blessing_channel_key)] = (
        model_blessing_resolver.task_info.name, model_blessing_channel_key)

    return [model_blessing_resolver, model_resolver]

  def _build_resolver_spec(self) -> List[pipeline_pb2.PipelineTaskSpec]:
    """Validates and builds ResolverSpec for this node.

    Returns:
      A list of PipelineTaskSpec represents the (potentially multiple) resolver
      task(s).
    Raises:
      TypeError: When get unsupported resolver policy. Currently only support
        LatestBlessedModelResolver and LatestArtifactsResolver.
    """
    assert isinstance(self._node, resolver.Resolver)

    if (self._exec_properties[resolver.RESOLVER_STRATEGY_CLASS] !=
        latest_blessed_model_resolver.LatestBlessedModelResolver and
        self._exec_properties[resolver.RESOLVER_STRATEGY_CLASS] !=
        latest_artifacts_resolver.LatestArtifactsResolver):
      raise TypeError(
          ('Unexpected resolver policy encountered. Currently '
           'only support latest artifact and latest blessed model '
           'resolver. Got: {}').format(
               self._exec_properties[resolver.RESOLVER_STRATEGY_CLASS]))
    if (self._exec_properties[resolver.RESOLVER_STRATEGY_CLASS] ==
        latest_blessed_model_resolver.LatestBlessedModelResolver):
      return self._build_latest_blessed_model_resolver()
    # Otherwise, this will be a LatestArtifactsResolver.
    return self._build_latest_artifact_resolver()
