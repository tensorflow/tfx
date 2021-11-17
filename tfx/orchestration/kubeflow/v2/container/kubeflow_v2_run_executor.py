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
"""Entrypoint for invoking TFX components in Kubeflow V2 runner."""

import argparse
import os
from typing import List, Tuple

from absl import app
from absl import logging
from absl.flags import argparse_flags
from kfp.pipeline_spec import pipeline_spec_pb2
from tfx.components.evaluator import executor as evaluator_executor
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.dsl.io import fileio
from tfx.orchestration.kubeflow.v2.container import kubeflow_v2_entrypoint_utils
from tfx.orchestration.portable import outputs_utils
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import import_utils

from google.protobuf import json_format


# TODO(b/166202742): Consolidate container entrypoint with Kubeflow runner.
# TODO(b/154046602): Consider put this function into tfx/orchestration, and
# unify the code paths to call into component executors.
def _run_executor(args: argparse.Namespace, beam_args: List[str]) -> None:
  """Selects a particular executor and run it based on name.

  Args:
    args:
      --executor_class_path: The import path of the executor class.
      --json_serialized_invocation_args: Full JSON-serialized parameters for
        this execution.
    beam_args: Optional parameter that maps to the optional_pipeline_args
      parameter in the pipeline, which provides additional configuration options
      for apache-beam and tensorflow.logging.
    For more about the beam arguments please refer to:
    https://cloud.google.com/dataflow/docs/guides/specifying-exec-params
  """
  logging.set_verbosity(logging.INFO)

  # Rehydrate inputs/outputs/exec_properties from the serialized metadata.
  executor_input = pipeline_spec_pb2.ExecutorInput()
  json_format.Parse(
      args.json_serialized_invocation_args,
      executor_input,
      ignore_unknown_fields=True)

  inputs_dict = executor_input.inputs.artifacts
  outputs_dict = executor_input.outputs.artifacts
  inputs_parameter = executor_input.inputs.parameters

  if fileio.exists(executor_input.outputs.output_file):
    # It has a driver that outputs the updated exec_properties in this file.
    with fileio.open(executor_input.outputs.output_file,
                     'rb') as output_meta_json:
      output_metadata = pipeline_spec_pb2.ExecutorOutput()
      json_format.Parse(
          output_meta_json.read(), output_metadata, ignore_unknown_fields=True)
      # Append/Overwrite exec_propertise.
      for k, v in output_metadata.parameters.items():
        inputs_parameter[k].CopyFrom(v)

  name_from_id = {}

  inputs = kubeflow_v2_entrypoint_utils.parse_raw_artifact_dict(
      inputs_dict, name_from_id)
  outputs = kubeflow_v2_entrypoint_utils.parse_raw_artifact_dict(
      outputs_dict, name_from_id)
  exec_properties = kubeflow_v2_entrypoint_utils.parse_execution_properties(
      inputs_parameter)
  logging.info('Executor %s do: inputs: %s, outputs: %s, exec_properties: %s',
               args.executor_class_path, inputs, outputs, exec_properties)
  executor_cls = import_utils.import_class_by_path(args.executor_class_path)
  if issubclass(executor_cls, base_beam_executor.BaseBeamExecutor):
    executor_context = base_beam_executor.BaseBeamExecutor.Context(
        beam_pipeline_args=beam_args, unique_id='')
  else:
    executor_context = base_executor.BaseExecutor.Context(
        extra_flags=beam_args, unique_id='')
  executor = executor_cls(executor_context)
  logging.info('Starting executor')
  executor.Do(inputs, outputs, exec_properties)

  # TODO(b/182316162): Unify publisher handling so that post-execution artifact
  # logic is more cleanly handled.
  outputs_utils.tag_output_artifacts_with_version(outputs)  # pylint: disable=protected-access

  # TODO(b/169583143): Remove this workaround when TFX migrates to use str-typed
  # id/name to identify artifacts.
  # Convert ModelBlessing artifact to use managed MLMD resource name.
  if (issubclass(executor_cls, evaluator_executor.Executor) and
      standard_component_specs.BLESSING_KEY in outputs):
    # Parse the parent prefix for managed MLMD resource name.
    kubeflow_v2_entrypoint_utils.refactor_model_blessing(
        artifact_utils.get_single_instance(
            outputs[standard_component_specs.BLESSING_KEY]), name_from_id)

  # Log the output metadata to a file. So that it can be picked up by MP.
  metadata_uri = executor_input.outputs.output_file
  executor_output = pipeline_spec_pb2.ExecutorOutput()
  for k, v in kubeflow_v2_entrypoint_utils.translate_executor_output(
      outputs, name_from_id).items():
    executor_output.artifacts[k].CopyFrom(v)

  fileio.makedirs(os.path.dirname(metadata_uri))
  with fileio.open(metadata_uri, 'wb') as f:
    f.write(json_format.MessageToJson(executor_output))


def _parse_flags(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Parses command line arguments.

  Args:
    argv: Unparsed arguments for run_executor.py. Known argument names include
      --executor_class_path: Python class of executor in format of
        <module>.<class>.
      --json_serialized_invocation_args: Full JSON-serialized parameters for
        this execution.

      The remaining part of the arguments will be parsed as the beam args used
      by each component executors. Some commonly used beam args are as follows:
        --runner: The beam pipeline runner environment. Can be DirectRunner (for
          running locally) or DataflowRunner (for running on GCP Dataflow
          service).
        --project: The GCP project ID. Neede when runner==DataflowRunner
        --direct_num_workers: Number of threads or subprocesses executing the
          work load.
      For more about the beam arguments please refer to:
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params

  Returns:
    Tuple of an argparse result and remaining beam args.
  """

  parser = argparse_flags.ArgumentParser()
  parser.add_argument(
      '--executor_class_path',
      type=str,
      required=True,
      help='Python class of executor in format of <module>.<class>.')
  parser.add_argument(
      '--json_serialized_invocation_args',
      type=str,
      required=True,
      help='JSON-serialized metadata for this execution.')

  return parser.parse_known_args(argv)


def main(parsed_argv: Tuple[argparse.Namespace, List[str]]):
  args, beam_args = parsed_argv
  _run_executor(args, beam_args)


if __name__ == '__main__':
  app.run(main, flags_parser=_parse_flags)
