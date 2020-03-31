# Lint as: python2, python3
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
"""Entrypoint for invoking TFX components in CAIP managed pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
from typing import List, Text
import absl

import tensorflow as tf
from google.protobuf import json_format
from tensorflow.python.platform import app  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_executor
from tfx.proto.orchestration import execution_result_pb2
from tfx.scripts import ai_platform_entrypoint_utils
from tfx.types import artifact_utils
from tfx.utils import import_utils


def _run_executor(args: argparse.Namespace, beam_args: List[Text]) -> None:
  """Selects a particular executor and run it based on name.

  Args:
    args:
      --executor_class_path: The import path of the executor class.
      --json_serialized_metadata: Full JSON-serialized metadata for this
        execution. See go/mp-alpha-placeholder for details.
    beam_args: Optional parameter that maps to the optional_pipeline_args
      parameter in the pipeline, which provides additional configuration options
      for apache-beam and tensorflow.logging.
    For more about the beam arguments please refer to:
    https://cloud.google.com/dataflow/docs/guides/specifying-exec-params
  """
  absl.logging.set_verbosity(absl.logging.INFO)

  # Rehydrate inputs/outputs/exec_properties from the serialized metadata.
  full_metadata_dict = json.loads(args.json_serialized_metadata)

  inputs_dict = full_metadata_dict['inputs']
  outputs_dict = full_metadata_dict['outputs']
  exec_properties_dict = full_metadata_dict['execution_properties']

  inputs = ai_platform_entrypoint_utils.parse_raw_artifact_dict(inputs_dict)
  outputs = ai_platform_entrypoint_utils.parse_raw_artifact_dict(outputs_dict)
  exec_properties = ai_platform_entrypoint_utils.parse_execution_properties(
      exec_properties_dict)
  absl.logging.info(
      'Executor %s do: inputs: %s, outputs: %s, exec_properties: %s' % (
          args.executor_class_path, inputs, outputs, exec_properties))
  executor_cls = import_utils.import_class_by_path(args.executor_class_path)
  executor_context = base_executor.BaseExecutor.Context(
      beam_pipeline_args=beam_args, unique_id='')
  executor = executor_cls(executor_context)
  absl.logging.info('Starting executor')
  executor.Do(inputs, outputs, exec_properties)

  # Log the output metadata to a file. So that it can be picked up by MP.
  metadata_uri = full_metadata_dict['output_metadata_uri']
  output_metadata = execution_result_pb2.ExecutorOutput()
  for key, output_artifacts in outputs.items():
    # Assuming each output is a singleton artifact.
    output_metadata.output_dict[key].CopyFrom(
        artifact_utils.get_single_instance(output_artifacts).mlmd_artifact)

  tf.io.gfile.GFile(metadata_uri,
                    'wb').write(json_format.MessageToJson(output_metadata))


def main(argv):
  """Parses the arguments for _run_executor() then invokes it.

  Args:
    argv: Unparsed arguments for run_executor.py. Known argument names include
      --executor_class_path: Python class of executor in format of
        <module>.<class>.
      --json_serialized_metadata: Full JSON-serialized metadata for this
        execution. See go/mp-alpha-placeholder for details.
      The remaining part of the arguments will be parsed as the beam args used
      by each component executors. Some commonly used beam args are as follows:
      --runner: The beam pipeline runner environment. Can be DirectRunner (for
        running locally) or DataflowRunner (for running on GCP Dataflow
        service).
      --project: The GCP project ID. Neede when runner==DataflowRunner
      --direct_num_workers: Number of threads or subprocesses executing the work
        load.
      For more about the beam arguments please refer to:
      https://cloud.google.com/dataflow/docs/guides/specifying-exec-params

  Returns:
    None

  Raises:
    None
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--executor_class_path',
      type=str,
      required=True,
      help='Python class of executor in format of <module>.<class>.')
  parser.add_argument(
      '--json_serialized_metadata',
      type=str,
      required=True,
      help='JSON-serialized metadata for this execution. '
      'See go/mp-alpha-placeholder for details.')

  args, beam_args = parser.parse_known_args(argv)
  _run_executor(args, beam_args)


if __name__ == '__main__':
  app.run(main=main)
