# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driver for file-based ExampleGen components in Kubeflow V2 runner."""

import argparse
import os
from typing import Any, Dict, List, Optional

from absl import app
from absl import logging
from absl.flags import argparse_flags
from kfp.pipeline_spec import pipeline_spec_pb2
from tfx.components.example_gen import driver
from tfx.components.example_gen import input_processor
from tfx.components.example_gen import utils
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2.container import kubeflow_v2_entrypoint_utils
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import proto_utils

from google.protobuf import json_format


def _run_driver(exec_properties: Dict[str, Any],
                outputs_dict: Dict[str, List[artifact.Artifact]],
                output_metadata_uri: str,
                name_from_id: Optional[Dict[int, str]] = None) -> None:
  """Runs the driver, writing its output as a ExecutorOutput proto.

  The main goal of this driver is to calculate the span and fingerprint of input
  data, allowing for the executor invocation to be skipped if the ExampleGen
  component has been previously run on the same data with the same
  configuration. This span and fingerprint are added as new custom execution
  properties to an ExecutorOutput proto and written to a GCS path. The CAIP
  pipelines system reads this file and updates MLMD with the new execution
  properties.


  Args:
    exec_properties:
      These are required to contain the following properties:
      'input_base_uri': A path from which files will be read and their
        span/fingerprint calculated.
      'input_config': A json-serialized tfx.proto.example_gen_pb2.InputConfig
        proto message.
        See https://www.tensorflow.org/tfx/guide/examplegen for more details.
      'output_config': A json-serialized tfx.proto.example_gen_pb2.OutputConfig
        proto message.
        See https://www.tensorflow.org/tfx/guide/examplegen for more details.
    outputs_dict: The mapping of the output artifacts.
    output_metadata_uri: A path at which an ExecutorOutput message will be
      written with updated execution properties and output artifacts. The CAIP
      Pipelines service will update the task's properties and artifacts prior to
      running the executor.
    name_from_id: Optional. Mapping from the converted int-typed id to str-typed
      runtime artifact name, which should be unique.
  """
  if name_from_id is None:
    name_from_id = {}

  logging.set_verbosity(logging.INFO)
  logging.info('exec_properties = %s\noutput_metadata_uri = %s',
               exec_properties, output_metadata_uri)

  input_base_uri = exec_properties.get(standard_component_specs.INPUT_BASE_KEY)

  input_config = example_gen_pb2.Input()
  proto_utils.json_to_proto(
      exec_properties[standard_component_specs.INPUT_CONFIG_KEY], input_config)

  range_config = None
  range_config_entry = exec_properties.get(
      standard_component_specs.RANGE_CONFIG_KEY)
  if range_config_entry:
    range_config = range_config_pb2.RangeConfig()
    proto_utils.json_to_proto(range_config_entry, range_config)

  processor = input_processor.FileBasedInputProcessor(input_base_uri,
                                                      input_config.splits,
                                                      range_config)
  span, version = processor.resolve_span_and_version()
  fingerprint = processor.get_input_fingerprint(span, version)

  logging.info('Calculated span: %s', span)
  logging.info('Calculated fingerprint: %s', fingerprint)

  exec_properties[utils.SPAN_PROPERTY_NAME] = span
  exec_properties[utils.FINGERPRINT_PROPERTY_NAME] = fingerprint
  exec_properties[utils.VERSION_PROPERTY_NAME] = version

  # Updates the input_config.splits.pattern.
  for split in input_config.splits:
    split.pattern = processor.get_pattern_for_span_version(
        split.pattern, span, version)
  exec_properties[standard_component_specs
                  .INPUT_CONFIG_KEY] = proto_utils.proto_to_json(input_config)

  if standard_component_specs.EXAMPLES_KEY not in outputs_dict:
    raise ValueError('Example artifact was missing in the ExampleGen outputs.')
  example_artifact = artifact_utils.get_single_instance(
      outputs_dict[standard_component_specs.EXAMPLES_KEY])

  driver.update_output_artifact(
      exec_properties=exec_properties,
      output_artifact=example_artifact.mlmd_artifact)

  # Log the output metadata file
  output_metadata = pipeline_spec_pb2.ExecutorOutput()
  output_metadata.parameters[
      utils.FINGERPRINT_PROPERTY_NAME].string_value = fingerprint
  output_metadata.parameters[utils.SPAN_PROPERTY_NAME].string_value = str(span)
  output_metadata.parameters[
      standard_component_specs
      .INPUT_CONFIG_KEY].string_value = json_format.MessageToJson(input_config)
  output_metadata.artifacts[
      standard_component_specs.EXAMPLES_KEY].artifacts.add().CopyFrom(
          kubeflow_v2_entrypoint_utils.to_runtime_artifact(
              example_artifact, name_from_id))

  fileio.makedirs(os.path.dirname(output_metadata_uri))
  with fileio.open(output_metadata_uri, 'wb') as f:
    f.write(json_format.MessageToJson(output_metadata, sort_keys=True))


def _parse_flags(argv: List[str]) -> argparse.Namespace:
  """Command lines flag parsing."""
  parser = argparse_flags.ArgumentParser()
  parser.add_argument(
      '--json_serialized_invocation_args',
      type=str,
      required=True,
      help='JSON-serialized metadata for this execution.')
  # Ignore unknown args which is expected. Beam related args are also supplied
  # as command line arguments.
  # TODO(b/182333035): Wrap beam related flags into a dedicated flag.
  namespace, _ = parser.parse_known_args(argv)
  return namespace


def main(args):
  executor_input = pipeline_spec_pb2.ExecutorInput()
  json_format.Parse(
      args.json_serialized_invocation_args,
      executor_input,
      ignore_unknown_fields=True)

  name_from_id = {}

  exec_properties = kubeflow_v2_entrypoint_utils.parse_execution_properties(
      executor_input.inputs.parameters)
  outputs_dict = kubeflow_v2_entrypoint_utils.parse_raw_artifact_dict(
      executor_input.outputs.artifacts, name_from_id)

  _run_driver(exec_properties, outputs_dict, executor_input.outputs.output_file,
              name_from_id)


if __name__ == '__main__':
  app.run(main, flags_parser=_parse_flags)
