# Lint as: python3
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
"""Enable KFP components to be used in a TFX pipeline."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Text
import warnings

from ml_metadata.proto import metadata_store_pb2
from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import placeholders
from tfx.types import artifact_utils


def _create_tfx_component_instance(
    component_spec,
    arguments: Dict[Text, Any],
    **kwargs
):
  """Construct a BaseComponent instance from KFP's ComponentSpec + arguments."""

  # Importing here to prevent import errors
  from kfp import components  # pylint: disable=g-import-not-at-top

  del kwargs

  # Replacing the placeholders for input and output paths with TFX placeholders
  def input_path_generator(input_name):
    return placeholders.InputUriPlaceholder(input_name=input_name)

  def output_path_generator(output_name):
    return placeholders.OutputUriPlaceholder(output_name=output_name)

  # Resolving the command-line
  resolved_cmd = components._components._resolve_command_line_and_paths(  # pylint: disable=protected-access
      component_spec=component_spec,
      arguments=arguments,
      input_path_generator=input_path_generator,
      output_path_generator=output_path_generator,
  )
  if resolved_cmd.input_paths or resolved_cmd.output_paths:
    warnings.warn('TFX does not have support for data passing yet. b/150670779'
                  'The component can fail at runtime.')

  tfx_input_specs = {}
  tfx_output_specs = {}

  for input_spec in component_spec.inputs or []:
    atrifact_class = _type_spec_to_artifact_class(input_spec.type)
    if input_spec.name not in resolved_cmd.inputs_consumed_by_value:
      tfx_input_specs[input_spec.name] = atrifact_class

  for output_spec in component_spec.outputs or []:
    atrifact_class = _type_spec_to_artifact_class(output_spec.type)
    tfx_input_specs[output_spec.name] = atrifact_class

  tfx_component = container_component.create_container_component(
      name=component_spec.name,
      image=component_spec.implementation.container.image,
      inputs=tfx_input_specs,
      outputs=tfx_output_specs,
      command=resolved_cmd.command + resolved_cmd.args,
  )
  tfx_component_instance = tfx_component(arguments)
  return tfx_component_instance


def enable_kfp_components():
  """Enables using KFP components in TFX SDK.

  All instantiated KFP components will become TFX component instances so they
  can be added to a TFX pipeline.

  Example:

    # Load/create any KFP components and build a pipeline
    import kfp
    my_trainer_op = kfp.components.load_component_from_url(...)
    my_predict_op = kfp.components.create_component_from_func(...)

    def build_my_pipeline():
      my_trainer = my_trainer_op(...)
      my_predictor = my_predict_op(my_trainer.outputs['model'])

      return [my_trainer, my_predictor]

    # Construct and run a TFX pipeline:
    kfp_components.enable_kfp_components()

    my_pipeline = pipeline.Pipeline(
      components=build_my_pipeline())

    KubeflowDagRunner(my_pipeline).run()
  """
  # Importing here to prevent import errors
  from kfp import components  # pylint: disable=g-import-not-at-top

  # Installing the component instantiation hook
  components._components._container_task_constructor = _create_tfx_component_instance  # pylint: disable=protected-access


def _type_spec_to_artifact_class(type_spec):
  artifact_type_name = str(type_spec) or 'Any'
  artifact_type = metadata_store_pb2.ArtifactType(name=artifact_type_name)
  return artifact_utils.get_artifact_type_class(artifact_type)
