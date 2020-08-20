# Lint as: python2, python3
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

from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import app  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.utils import io_utils

_LOCAL_INFERENCE_TIMEOUT_SECONDS = 5.0

_LABEL_KEY = 'tips'


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _make_proto_coder(schema):
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.ExampleProtoCoder(raw_schema)


def _make_csv_coder(schema, column_names):
  """Return a coder for tf.transform to read csv files."""
  raw_feature_spec = _get_raw_feature_spec(schema)
  parsing_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  return tft_coders.CsvCoder(column_names, parsing_schema)


def _read_schema(path):
  """Reads a schema from the provided location.

  Args:
    path: The location of the file holding a serialized Schema proto.

  Returns:
    An instance of Schema or None if the input argument is None
  """
  result = schema_pb2.Schema()
  contents = file_io.read_file_to_string(path)
  text_format.Parse(contents, result)
  return result


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


def _do_aiplatform_inference(model, version, serialized_examples):
  """Performs inference on the model:version in AI Platform."""
  working_dir = tempfile.mkdtemp()
  instances_file = os.path.join(working_dir, 'test.json')
  json_examples = []
  for serialized_example in serialized_examples:
    # The encoding follows the example in:
    # https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/quests/tpu/invoke_model.py
    json_examples.append('{ "inputs": { "b64": "%s" } }' %
                         base64.b64encode(serialized_example).decode('utf-8'))
  file_io.write_string_to_file(instances_file, '\n'.join(json_examples))
  gcloud_command = [
      'gcloud', 'ai-platform', 'predict', '--model', model, '--version',
      version, '--json-instances', instances_file
  ]
  print(subprocess.check_output(gcloud_command))


def _do_inference(model_handle, examples_file, num_examples, schema):
  """Sends requests to the model and prints the results.

  Args:
    model_handle: handle to the model. This can be either
     "aiplatform:model:version" or "host:port"
    examples_file: path to csv file containing examples, with the first line
      assumed to have the column headers
    num_examples: number of requests to send to the server
    schema: a Schema describing the input data

  Returns:
    Response from model server
  """
  filtered_features = [
      feature for feature in schema.feature if feature.name != _LABEL_KEY
  ]
  del schema.feature[:]
  schema.feature.extend(filtered_features)

  column_names = io_utils.load_csv_column_names(examples_file)
  csv_coder = _make_csv_coder(schema, column_names)
  proto_coder = _make_proto_coder(schema)

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
  if parsed_model_handle[0] == 'aiplatform':
    _do_aiplatform_inference(
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
      help=('Prediction service host:port or aiplatform:model:version'),
      required=True)

  parser.add_argument(
      '--examples_file',
      help=('Path to csv file containing examples.'),
      required=True)

  parser.add_argument(
      '--schema_file', help='File holding the schema for the input data')
  known_args, _ = parser.parse_known_args()
  _do_inference(known_args.server, known_args.examples_file,
                known_args.num_examples, _read_schema(known_args.schema_file))


if __name__ == '__main__':
  app.run(main)
