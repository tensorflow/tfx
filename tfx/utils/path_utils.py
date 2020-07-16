# Lint as: python2, python3
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
"""Utilities for retrieving paths for various types of artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text
import absl

import tensorflow as tf

from tfx.utils import io_utils

EVAL_MODEL_DIR = 'eval_model_dir'
SERVING_MODEL_DIR = 'serving_model_dir'

"""Directory structure of exported model for estimator based trainer:

  |-- <ModelExportPath>
      |-- EVAL_MODEL_DIR  <- eval_model_dir, eval_model_path
          |-- saved_model.pb
          |-- ...
      |-- SERVING_MODEL_DIR  <- serving_model_dir, serving_model_path
          |-- saved_model.pb
          |-- ...

For generic trainer with Keras, there won't be eval model:
  |-- <ModelExportPath>
      |-- SERVING_MODEL_DIR  <- serving_model_dir, serving_model_path
          |-- saved_model.pb
          |-- ...

TODO(b/160795287): Deprecate estimator based executor.

Support for estimator-based executor and model export will be
deprecated soon. The following estimator working directory
structure is still supported for backwards compatibility:

Directory structure of exported model for estimator based trainer:
  |-- <ModelExportPath>
      |-- EVAL_MODEL_DIR  <- eval_model_dir
          |-- <timestamped model>  <- eval_model_path
              |-- saved_model.pb
              |-- ...
      |-- SERVING_MODEL_DIR  <- serving_model_dir
          |-- export
              |-- <exporter name>
                  |-- <timestamped model>  <- serving_model_path
                      |-- saved_model.pb
                      |-- ...
          |-- ...
"""


def eval_model_dir(output_uri: Text) -> Text:
  """Returns directory for exported model for evaluation purpose."""
  return os.path.join(output_uri, EVAL_MODEL_DIR)


def eval_model_path(output_uri: Text) -> Text:
  """Returns final path to exported model for evaluation purpose."""
  model_dir = eval_model_dir(output_uri)
  model_file = os.path.join(model_dir, 'saved_model.pb')
  if tf.io.gfile.exists(model_file):
    return model_dir
  elif tf.io.gfile.exists(model_dir):
    # TODO(b/160795287): Deprecate estimator based executor.
    absl.logging.warning('Support for estimator-based executor and model'
                         ' export will be deprecated soon. Please use'
                         ' export structure '
                         '<ModelExportPath>/eval_model_dir/saved_model.pb"')
    return io_utils.get_only_uri_in_dir(model_dir)
  else:
    # If eval model doesn't exist, use serving model for eval.
    return serving_model_path(output_uri)


def serving_model_dir(output_uri: Text) -> Text:
  """Returns directory for exported model for serving purpose."""
  return os.path.join(output_uri, SERVING_MODEL_DIR)


def serving_model_path(output_uri: Text) -> Text:
  """Returns path for exported serving model."""
  model_dir = serving_model_dir(output_uri)
  export_dir = os.path.join(model_dir, 'export')
  if tf.io.gfile.exists(export_dir):
    # TODO(b/160795287): Deprecate estimator based executor.
    absl.logging.warning(
        'Support for estimator-based executor and model export'
        ' will be deprecated soon. Please use export structure '
        '<ModelExportPath>/serving_model_dir/saved_model.pb"')
    model_dir = io_utils.get_only_uri_in_dir(export_dir)
    return io_utils.get_only_uri_in_dir(model_dir)
  else:
    # If dir doesn't match estimator structure, use serving model root directly.
    return model_dir
