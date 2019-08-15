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
"""Main entrypoint for containers with Kubeflow TFX component executors.

We cannot use the existing TFX container entrypoint for the following
reason:Say component A requires inputs from component B output O1 and
component C output O2. Now, the inputs to A is a serialized dictionary
contained O1 and O2. But we need Argo to combine O1 and O2 into the expected
dictionary of artifact/artifact_type types, which isn't possible. Hence, we
need each output from a component to be individual argo output parameters so
they can be passed into downstream components as input parameters via Argo.

TODO(ajaygopinathan): The input names below are hardcoded and can easily
diverge from the actual names and types expected by the underlying executors.
Look into how we can dynamically generate the required inputs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys
from tfx.orchestration.kubeflow import executor_wrappers as wrappers


def main():
  # Log to the container's stdout so Kubeflow Pipelines UI can display logs to
  # the user.
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--exec_properties', type=str, required=True)
  parser.add_argument('--outputs', type=str, required=True)
  parser.add_argument('--executor_class_path', type=str, required=True)

  subparsers = parser.add_subparsers(dest='executor')

  subparser = subparsers.add_parser('CsvExampleGen')
  subparser.add_argument('--input_base', type=str, required=True)
  subparser.set_defaults(executor=wrappers.CsvExampleGenWrapper)

  subparser = subparsers.add_parser('BigQueryExampleGen')
  subparser.set_defaults(executor=wrappers.BigQueryExampleGenWrapper)

  subparser = subparsers.add_parser('StatisticsGen')
  subparser.add_argument('--input_data', type=str, required=True)
  subparser.set_defaults(executor=wrappers.StatisticsGenWrapper)

  subparser = subparsers.add_parser('SchemaGen')
  subparser.add_argument('--stats', type=str, required=True)
  subparser.set_defaults(executor=wrappers.SchemaGenWrapper)

  subparser = subparsers.add_parser('ExampleValidator')
  subparser.add_argument('--stats', type=str, required=True)
  subparser.add_argument('--schema', type=str, required=True)
  subparser.set_defaults(executor=wrappers.ExampleValidatorWrapper)

  subparser = subparsers.add_parser('Transform')
  subparser.add_argument('--input_data', type=str, required=True)
  subparser.add_argument('--schema', type=str, required=True)
  subparser.set_defaults(executor=wrappers.TransformWrapper)

  subparser = subparsers.add_parser('Trainer')
  subparser.add_argument('--examples', type=str, required=True)
  subparser.add_argument('--transform_output', type=str, required=True)
  subparser.add_argument('--schema', type=str, required=True)
  subparser.set_defaults(executor=wrappers.TrainerWrapper)

  subparser = subparsers.add_parser('Evaluator')
  subparser.add_argument('--examples', type=str, required=True)
  subparser.add_argument('--model_exports', type=str, required=True)
  subparser.set_defaults(executor=wrappers.EvaluatorWrapper)

  subparser = subparsers.add_parser('ModelValidator')
  subparser.add_argument('--examples', type=str, required=True)
  subparser.add_argument('--model', type=str, required=True)
  subparser.set_defaults(executor=wrappers.ModelValidatorWrapper)

  subparser = subparsers.add_parser('Pusher')
  subparser.add_argument('--model_export', type=str, required=True)
  subparser.add_argument('--model_blessing', type=str, required=True)
  subparser.set_defaults(executor=wrappers.PusherWrapper)

  args = parser.parse_args()
  executor = args.executor(args)
  executor.run()


if __name__ == '__main__':
  main()
