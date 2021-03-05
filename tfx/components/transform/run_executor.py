# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Invoke transform executor for data transformation."""

import argparse

from typing import List, Tuple

import absl
from absl import app
from absl.flags import argparse_flags

from tfx.components.transform import labels
from tfx.components.transform.executor import Executor
from tfx.proto import example_gen_pb2


def _run_transform(args, beam_pipeline_args):
  """Construct and run transform executor."""
  absl.logging.set_verbosity(absl.logging.INFO)

  inputs = {
      labels.ANALYZE_DATA_PATHS_LABEL:
          args.analyze_examples,
      labels.ANALYZE_PATHS_FILE_FORMATS_LABEL: [labels.FORMAT_TFRECORD] *
                                               len(args.analyze_examples),
      labels.TRANSFORM_DATA_PATHS_LABEL: [
          args.analyze_examples + args.transform_only_examples
      ],
      labels.TRANSFORM_PATHS_FILE_FORMATS_LABEL:
          [labels.FORMAT_TFRECORD] *
          (len(args.analyze_examples) + len(args.transform_only_examples)),
      labels.SCHEMA_PATH_LABEL:
          args.input_schema_path,
      labels.PREPROCESSING_FN:
          args.preprocessing_fn_path,
      labels.EXAMPLES_DATA_FORMAT_LABEL:
          example_gen_pb2.PayloadFormat.Value(args.example_data_format),
      labels.COMPUTE_STATISTICS_LABEL:
          args.compute_statistics,
      labels.BEAM_PIPELINE_ARGS:
          beam_pipeline_args,
  }
  outputs = {
      labels.TRANSFORM_METADATA_OUTPUT_PATH_LABEL: args.transform_fn,
      labels.TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL: (
          args.transformed_examples),
      labels.PER_SET_STATS_OUTPUT_PATHS_LABEL: (args.per_set_stats_outputs),
      labels.TEMP_OUTPUT_LABEL: args.tmp_location,
  }
  executor = Executor(Executor.Context(beam_pipeline_args=beam_pipeline_args))
  executor.Transform(inputs, outputs, args.status_file)


def _parse_flags(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
  """Command lines flag parsing."""
  parser = argparse_flags.ArgumentParser()

  # Arguments in inputs
  parser.add_argument(
      '--input_schema_path',
      type=str,
      required=True,
      help='Path to input schema')
  parser.add_argument(
      '--preprocessing_fn_path',
      type=str,
      default='',
      required=True,
      help='Path to a preprocessing_fn module')
  parser.add_argument(
      '--use_tfdv',
      type=bool,
      default=True,
      help='Deprecated and ignored. DO NOT SET.')
  parser.add_argument(
      '--compute_statistics',
      type=bool,
      default=False,
      help='Whether computes statistics')
  parser.add_argument(
      '--analyze_examples',
      nargs='+',
      default='',
      type=str,
      help='A space-separated list of paths to examples to be analyzed '
      'and transformed')
  parser.add_argument(
      '--transform_only_examples',
      nargs='+',
      default='',
      type=str,
      help='A space-separated list of paths to examples to be transformed only')
  parser.add_argument(
      '--example_data_format',
      type=str,
      default=example_gen_pb2.PayloadFormat.Name(
          example_gen_pb2.FORMAT_TF_EXAMPLE),
      help='Example data format')
  # Arguments in outputs
  parser.add_argument(
      '--transform_fn',
      type=str,
      required=True,
      help='Path that TFTransformOutput will write to')
  parser.add_argument(
      '--tmp_location',
      type=str,
      required=True,
      help='Path to write temporary files. Executor does not own this '
      'directory. User or caller is responsible for cleanup')
  parser.add_argument(
      '--transformed_examples',
      nargs='+',
      type=str,
      default=[],
      help='A space-separated list of paths to write transformed examples')
  parser.add_argument(
      '--per_set_stats_outputs',
      nargs='+',
      type=str,
      default=[],
      help='Paths to statistics output')
  parser.add_argument(
      '--status_file', type=str, default='', help='Path to write status')
  return parser.parse_known_args(argv)


def main(parsed_argv: Tuple[argparse.Namespace, List[str]]):
  args, beam_args = parsed_argv
  _run_transform(args, beam_args)


if __name__ == '__main__':
  app.run(main, flags_parser=_parse_flags)
