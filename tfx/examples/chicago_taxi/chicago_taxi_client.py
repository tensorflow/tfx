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
"""A client for the chicago_taxi demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
import json
import os
import subprocess
import tempfile

import requests
import tensorflow as tf

from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tfx.examples.chicago_taxi.trainer import taxi

_LOCAL_INFERENCE_TIMEOUT_SECONDS = 5.0


def _do_local_inference(host, port, serialized_examples):
  """Performs inference on a model hosted by the host:port server."""

  json_examples = []
  for serialized_example in serialized_examples:
    # The encoding follows the guidelines in:
    # https://www.tensorflow.org/tfx/serving/api_rest
    example_bytes = base64.b64encode(serialized_example).decode('utf-8')
    predict_request = '{ "b64": "%s" }' % example_bytes
    json_examples.append(predict_request)

  json_request = '{ "instances": [' + ','.join(map(str, json_examples)) + ']}'

  server_url = 'http://' + host + ':' + port + '/v1/models/chicago_taxi:predict'
  response = requests.post(
      server_url, data=json_request, timeout=_LOCAL_INFERENCE_TIMEOUT_SECONDS)
  response.raise_for_status()
  prediction = response.json()
  print(json.dumps(prediction, indent=4))


def _do_mlengine_inference(model, version, serialized_examples):
  """Performs inference on the model:version in CMLE."""
  working_dir = tempfile.mkdtemp()
  instances_file = os.path.join(working_dir, 'test.json')
  json_examples = []
  for serialized_example in serialized_examples:
    # The encoding follows the example in:
    # https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/quests/tpu/invoke_model.py
    json_examples.append(
        '{ "inputs": { "b64": "%s" } }' % base64.b64encode(serialized_example))
  file_io.write_string_to_file(instances_file, '\n'.join(json_examples))
  gcloud_command = [
      'gcloud', 'ml-engine', 'predict', '--model', model, '--version', version,
      '--json-instances', instances_file
  ]
  print(subprocess.check_output(gcloud_command))


def _do_inference(model_handle, examples_file, num_examples, schema):
  """Sends requests to the model and prints the results.

  Args:
    model_handle: handle to the model. This can be either
     "mlengine:model:version" or "host:port"
    examples_file: path to csv file containing examples, with the first line
      assumed to have the column headers
    num_examples: number of requests to send to the server
    schema: a Schema describing the input data

  Returns:
    Response from model server
  """
  filtered_features = [
      feature for feature in schema.feature if feature.name != taxi.LABEL_KEY
  ]
  del schema.feature[:]
  schema.feature.extend(filtered_features)

  csv_coder = taxi.make_csv_coder(schema)
  proto_coder = taxi.make_proto_coder(schema)

  input_file = open(examples_file, 'r')
  input_file.readline()  # skip header line

  serialized_examples = []
  for _ in range(num_examples):
    one_line = input_file.readline()
    if not one_line:
      print('End of example file reached')
      break
    one_example = csv_coder.decode(one_line)

    serialized_example = proto_coder.encode(one_example)
    serialized_examples.append(serialized_example)

  parsed_model_handle = model_handle.split(':')
  if parsed_model_handle[0] == 'mlengine':
    _do_mlengine_inference(
        model=parsed_model_handle[1],
        version=parsed_model_handle[2],
        serialized_examples=serialized_examples)
  else:
    _do_local_inference(
        host=parsed_model_handle[0],
        port=parsed_model_handle[1],
        serialized_examples=serialized_examples)


def main(_):
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_examples',
      help=('Number of examples to send to the server.'),
      default=1,
      type=int)

  parser.add_argument(
      '--server',
      help=('Prediction service host:port or mlengine:model:version'),
      required=True)

  parser.add_argument(
      '--examples_file',
      help=('Path to csv file containing examples.'),
      required=True)

  parser.add_argument(
      '--schema_file', help='File holding the schema for the input data')
  known_args, _ = parser.parse_known_args()
  _do_inference(known_args.server,
                known_args.examples_file, known_args.num_examples,
                taxi.read_schema(known_args.schema_file))


if __name__ == '__main__':
  tf.app.run()
