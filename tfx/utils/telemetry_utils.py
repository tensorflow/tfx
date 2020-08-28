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
"""Utilities for gathering telemetry for TFX components and pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import re
import sys
import threading
from typing import Dict, List, Text

from tfx import version

# Common label names used.
LABEL_TFX_RUNNER = 'tfx_runner'
LABEL_TFX_EXECUTOR = 'tfx_executor'
_LABEL_TFX_VERSION = 'tfx_version'
_LABEL_TFX_PY_VERSION = 'tfx_py_version'

# The GKE pod label indicating the SDK environment.
LABEL_KFP_SDK_ENV = 'pipelines.kubeflow.org/pipeline-sdk-type'

# Thread local labels registered so far.
_thread_local_labels_state = threading.local()
_thread_local_labels_state.dictionary = {}


@contextlib.contextmanager
def scoped_labels(labels: Dict[Text, Text]):
  """Register thread local labels used in current context."""
  if getattr(_thread_local_labels_state, 'dictionary', None) is None:
    _thread_local_labels_state.dictionary = {}
  for key, value in labels.items():
    _thread_local_labels_state.dictionary[key] = _normalize_label(value)
  try:
    yield
  finally:
    for key in labels:
      _thread_local_labels_state.dictionary.pop(key)


def _normalize_label(value: Text) -> Text:
  """Lowercase and replace illegal characters in labels."""
  # See https://cloud.google.com/compute/docs/labeling-resources.
  return re.sub(r'[^a-z0-9\_\-]', '-', value.lower())[-63:]


def get_labels_dict() -> Dict[Text, Text]:
  """Get all registered and system generated labels as a dict.

  Returns:
    All registered and system generated labels as a dict.
  """
  result = dict(
      {
          _LABEL_TFX_VERSION:
              version.__version__,
          _LABEL_TFX_PY_VERSION:
              '%d.%d' % (sys.version_info.major, sys.version_info.minor),
      }, **getattr(_thread_local_labels_state, 'dictionary', {}))
  for k, v in result.items():
    result[k] = _normalize_label(v)
  return result


def make_beam_labels_args() -> List[Text]:
  """Make Beam arguments for common labels used in TFX pipelines.

  Returns:
    New Beam pipeline args with labels.
  """
  labels = get_labels_dict()
  # See following file for reference to the '--labes ' flag.
  # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py
  result = []
  for k in sorted(labels):
    result.extend(['--labels', '%s=%s' % (k, labels[k])])
  return result
