# Lint as: python2, python3
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
"""Utility functions for cleaning directory structure"""

import absl
from typing import Text

from tfx.utils import io_utils
from tfx.utils import path_utils

def copy_model(working_dir: Text, dest: Text, tag: Text) -> None:
  """Copy a specified model from working dir to specified destination."""
  path_fn = None
  if tag == 'serving':
    path_fn = path_utils.serving_model_working_path
  elif tag == 'eval':
    path_fn = path_utils.eval_model_working_path
  else:
    raise ValueError('Invalid input tag: {}.'.format(tag))

  source = path_fn(working_dir)
  io_utils.copy_dir(source, dest)
  absl.logging.info('%s model copied to: %s.', tag.capitalize(), dest)
