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
"""Main entrypoint for containers with TF serving."""

import argparse
import logging
import os
import sys

from tfx.proto import pusher_pb2
from tfx.utils import io_utils

_DEFAULT_MODEL_NAME = 'default'
_SERVING_DIR = '/models'

def main():
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('--model_uri', type=str, required=True)
  parser.add_argument('--model_name', type=str, default=_DEFAULT_MODEL_NAME)

  args = parser.parse_args()

  # download the model files to local serving directory
  io_utils.copy_dir(args.model_uri, os.path.join(_SERVING_DIR, args.model_name))


if __name__ == '__main__':
  main()
