# Copyright 2019 Google LLC
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
"""Common script to invoke TFX executors."""

import argparse
import base64
import json
from typing import List, Tuple

import absl
from absl import app
from absl.flags import argparse_flags
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import import_utils


def _run_executor(args, pipeline_args) -> None:
  r"""Select a particular executor and run it based on name.

  # pylint: disable=line-too-long
  _run_executor() is used to invoke a class subclassing
  tfx.dsl.components.base.base_executor.BaseExecutor.  This function can be used for
  both invoking the executor on remote environments as well as for unit testing
  of executors.

  How to invoke an executor as standalone:
  # TODO(b/132958430): Create utility script to generate arguments for run_executor.py
  First, the input data needs to be prepared.  An easy way to generate the test
  data is to fully run the pipeline once.  This will generate the data to be
  used for testing as well as log the artifacts to be used as input parameters.
  In each executed component, three log entries will be generated similar to the
  below:
  ```
  [2019-05-16 08:59:27,117] {logging_mixin.py:95} INFO - [2019-05-16 08:59:27,116] {base_executor.py:72} INFO - Starting Executor execution.
  [2019-05-16 08:59:27,117] {logging_mixin.py:95} INFO - [2019-05-16 08:59:27,117] {base_executor.py:74} INFO - Inputs for Executor is: {"input_base": [{"artifact": {"id": "1", "typeId": "1", "uri": "/usr/local/google/home/khaas/taxi/data/simple", "properties": {"split": {"stringValue": ""}, "state": {"stringValue": "published"}, "span": {"intValue": "1"}, "type_name": {"stringValue": "ExternalPath"}}}, "artifact_type": {"id": "1", "name": "ExternalPath", "properties": {"span": "INT", "name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING"}}}]}
  [2019-05-16 08:59:27,117] {logging_mixin.py:95} INFO - [2019-05-16 08:59:27,117] {base_executor.py:76} INFO - Outputs for Executor is: {"examples": [{"artifact": {"uri": "/usr/local/google/home/khaas/tfx/pipelines/chicago_taxi_simple/CsvExampleGen/examples/1/train/", "properties": {"type_name": {"stringValue": "ExamplesPath"}, "split": {"stringValue": "train"}, "span": {"intValue": "1"}}}, "artifact_type": {"name": "ExamplesPath", "properties": {"name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING", "span": "INT"}}}, {"artifact": {"uri": "/usr/local/google/home/khaas/tfx/pipelines/chicago_taxi_simple/CsvExampleGen/examples/1/eval/", "properties": {"type_name": {"stringValue": "ExamplesPath"}, "split": {"stringValue": "eval"}, "span": {"intValue": "1"}}}, "artifact_type": {"name": "ExamplesPath", "properties": {"name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING", "span": "INT"}}}]}
  [2019-05-16 08:59:27,117] {logging_mixin.py:95} INFO - [2019-05-16 08:59:27,117] {base_executor.py:78} INFO - Execution properties for Executor is: {"output": "{  \"splitConfig\": {\"splits\": [{\"name\": \"train\", \"hashBuckets\": 2}, {\"name\": \"eval\",\"hashBuckets\": 1}]}}"}
  ```
  Each of these map directly to the input parameters expected by run_executor():
  ```
  python scripts/run_executor.py \
      --executor_class_path=tfx.components.example_gen.csv_example_gen.executor.Executor \
      --inputs={"input_base": [{"artifact": {"id": "1", "typeId": "1", "uri": "/usr/local/google/home/khaas/taxi/data/simple", "properties": {"split": {"stringValue": ""}, "state": {"stringValue": "published"}, "span": {"intValue": "1"}, "type_name": {"stringValue": "ExternalPath"}}}, "artifact_type": {"id": "1", "name": "ExternalPath", "properties": {"span": "INT", "name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING"}}}]} \
      --outputs={"examples": [{"artifact": {"uri": "/usr/local/google/home/khaas/tfx/pipelines/chicago_taxi_simple/CsvExampleGen/examples/1/train/", "properties": {"type_name": {"stringValue": "ExamplesPath"}, "split": {"stringValue": "train"}, "span": {"intValue": "1"}}}, "artifact_type": {"name": "ExamplesPath", "properties": {"name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING", "span": "INT"}}}, {"artifact": {"uri": "/usr/local/google/home/khaas/tfx/pipelines/chicago_taxi_simple/CsvExampleGen/examples/1/eval/", "properties": {"type_name": {"stringValue": "ExamplesPath"}, "split": {"stringValue": "eval"}, "span": {"intValue": "1"}}}, "artifact_type": {"name": "ExamplesPath", "properties": {"name": "STRING", "type_name": "STRING", "split": "STRING", "state": "STRING", "span": "INT"}}}]} \
      --exec-properties={"output": "{  \"splitConfig\": {\"splits\": [{\"name\": \"train\", \"hashBuckets\": 2}, {\"name\": \"eval\",\"hashBuckets\": 1}]}}"}
  ```
  # pylint: disable=line-too-long

  Args:
    args:
      - inputs: The input artifacts for this execution, serialized as JSON.
      - outputs: The output artifacts to be generated by this execution,
        serialized as JSON.
      - exec_properties: The execution properties to be used by this execution,
        serialized as JSON.
    pipeline_args: Optional parameter that maps to the optional_pipeline_args
    parameter in the pipeline, which provides additional configuration options
    for apache-beam and tensorflow.logging.

  Returns:
    None

  Raises:
    None
  """

  absl.logging.set_verbosity(absl.logging.INFO)

  (inputs_str, outputs_str,
   exec_properties_str) = (args.inputs or base64.b64decode(args.inputs_base64),
                           args.outputs or
                           base64.b64decode(args.outputs_base64),
                           args.exec_properties or
                           base64.b64decode(args.exec_properties_base64))

  inputs = artifact_utils.parse_artifact_dict(inputs_str)
  outputs = artifact_utils.parse_artifact_dict(outputs_str)
  exec_properties = json.loads(exec_properties_str)
  absl.logging.info(
      'Executor {} do: inputs: {}, outputs: {}, exec_properties: {}'.format(
          args.executor_class_path, inputs, outputs, exec_properties))
  executor_cls = import_utils.import_class_by_path(args.executor_class_path)
  if issubclass(executor_cls,
                base_beam_executor.BaseBeamExecutor):
    executor_context = base_beam_executor.BaseBeamExecutor.Context(
        beam_pipeline_args=pipeline_args,
        tmp_dir=args.temp_directory_path,
        unique_id='')
  else:
    executor_context = base_executor.BaseExecutor.Context(
        extra_flags=pipeline_args,
        tmp_dir=args.temp_directory_path,
        unique_id='')
  executor = executor_cls(executor_context)
  absl.logging.info('Starting executor')
  executor.Do(inputs, outputs, exec_properties)

  # The last line of stdout will be pushed to xcom by Airflow.
  if args.write_outputs_stdout:
    print(artifact_utils.jsonify_artifact_dict(outputs))


def _parse_flags(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Parses command line arguments.

  # pylint: disable=line-too-long
  Args:
    argv: Unparsed arguments for run_executor.py
      --executor_class_path: Python class of executor in format of <module>.<class>.
      --temp_directory_path: Common temp directory path for executors.
      --inputs: JSON serialized dict of input artifacts.  If the input needs to be base64-encoded, use --inputs-base64 instead.
      --inputs-base64: base64-encoded JSON serialized dict of input artifacts.  If the input is not base64-encoded, use --inputs instead.
      --outputs: JSON serialized dict of output artifacts.  If the output needs to be base64-encoded, use --outputs-base64 instead.
      --outputs-base64: base64-encoded JSON serialized dict of output artifacts.  If the output is not base64-encoded, use --outputs instead.
      --exec_properties: JSON serialized dict of (non artifact) execution properties.  If the execution properties need to be base64-encoded, use --exec_properties-base64 instead.
      --exec_properties-base64: base64-encoded JSON serialized dict of (non artifact) execution properties.  If the execution properties are not base64-encoded, use --exec_properties instead.
      --write_outputs_stdout: Write outputs to last line of stdout, which will be pushed to xcom in Airflow. Please ignore by other users or orchestrators.
  # pylint: disable=line-too-long

  Returns:
    None

  Raises:
    None
  """

  parser = argparse_flags.ArgumentParser()
  parser.add_argument(
      '--executor_class_path',
      type=str,
      required=True,
      help='Python class of executor in format of <module>.<class>.')
  parser.add_argument(
      '--temp_directory_path',
      type=str,
      help='common temp directory path for executors')
  inputs_group = parser.add_mutually_exclusive_group(required=True)
  inputs_group.add_argument(
      '--inputs',
      type=str,
      help='json serialized dict of input artifacts.')
  inputs_group.add_argument(
      '--inputs-base64',
      type=str,
      help='base64 encoded json serialized dict of input artifacts.')

  outputs_group = parser.add_mutually_exclusive_group(required=True)
  outputs_group.add_argument(
      '--outputs',
      type=str,
      help='json serialized dict of output artifacts.')
  outputs_group.add_argument(
      '--outputs-base64',
      type=str,
      help='base64 encoded json serialized dict of output artifacts.')

  execution_group = parser.add_mutually_exclusive_group(required=True)
  execution_group.add_argument(
      '--exec-properties',
      type=str,
      help='json serialized dict of (non artifact) execution properties.')
  execution_group.add_argument(
      '--exec-properties-base64',
      type=str,
      help='json serialized dict of (non artifact) execution properties.')

  parser.add_argument(
      '--write-outputs-stdout',
      dest='write_outputs_stdout',
      action='store_true',
      help='Write outputs to last line of stdout, which will '
      'be pushed to xcom in Airflow. Please ignore by other users or '
      'orchestrators.')

  return parser.parse_known_args(argv)


def main(parsed_argv: Tuple[argparse.Namespace, List[str]]):
  args, beam_args = parsed_argv
  _run_executor(args, beam_args)


if __name__ == '__main__':
  app.run(main, flags_parser=_parse_flags)

