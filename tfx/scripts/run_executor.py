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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import json
import tensorflow as tf
from tensorflow.python.platform import app  # pylint: disable=g-direct-tensorflow-import
from tfx.utils import import_utils
from tfx.utils import types


def _run_executor(args, pipeline_args):
  """Select a particular executor and run it based on name."""
  tf.logging.set_verbosity(tf.logging.INFO)

  (inputs_str, outputs_str,
   exec_properties_str) = (args.inputs or base64.b64decode(args.inputs_base64),
                           args.outputs or
                           base64.b64decode(args.outputs_base64),
                           args.exec_properties or
                           base64.b64decode(args.exec_properties_base64))

  inputs = types.parse_tfx_type_dict(inputs_str)
  outputs = types.parse_tfx_type_dict(outputs_str)
  exec_properties = json.loads(exec_properties_str)
  tf.logging.info(
      'Executor {} do: inputs: {}, outputs: {}, exec_properties: {}'.format(
          args.executor, inputs, outputs, exec_properties))

  executor_cls = import_utils.import_class_by_path(args.executor_class_path)
  executor = executor_cls(beam_pipeline_args=pipeline_args)
  tf.logging.info('Starting executor')
  executor.Do(inputs, outputs, exec_properties)

  # The last line of stdout will be pushed to xcom by Airflow.
  if args.write_outputs_stdout:
    print(types.jsonify_tfx_type_dict(outputs))


def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--executor_class_path',
      type=str,
      required=True,
      help='Python class of executor in format of <module>.<class>.')
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

  args, beam_pipeline_args = parser.parse_known_args(argv)
  _run_executor(args, beam_pipeline_args)


if __name__ == '__main__':
  app.run(main=main)
