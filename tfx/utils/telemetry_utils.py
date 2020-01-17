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

import re
import sys
from typing import Dict, List, Text

from tfx import version

# Common label names used.
_TFX_EXECUTOR = 'tfx_executor'
_TFX_VERSION = 'tfx_version'
_TFX_PY_VERSION = 'tfx_py_version'


def _normalize_label(value: Text) -> Text:
  """Lowercase and replace illegal characters in labels."""
  # See https://cloud.google.com/compute/docs/labeling-resources.
  return re.sub(r'[^a-z0-9\_\-]', '-', value.lower())[-63:]


def get_labels_dict(tfx_executor: Text) -> Dict[Text, Text]:
  """Get all registered and system generated labels as a dict.

  Args:
    tfx_executor: Executor path of TFX.

  Returns:
    All registered and system generated labels as a dict.
  """
  result = dict({
      _TFX_VERSION:
          version.__version__,
      _TFX_PY_VERSION:
          '%d.%d' % (sys.version_info.major, sys.version_info.minor),
      _TFX_EXECUTOR:
          tfx_executor,
  })
  for k, v in result.items():
    result[k] = _normalize_label(v)
  return result


def make_beam_labels_args(tfx_executor: Text) -> List[Text]:
  """Make Beam arguments for common labels used in TFX pipelines.

  Args:
    tfx_executor: Executor name of TFX.

  Returns:
    New Beam pipeline args with labels.
  """
  labels = get_labels_dict(tfx_executor)
  # See following file for reference to the '--labes ' flag.
  # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py
  result = []
  for k in sorted(labels):
    result.extend(['--labels', '%s=%s' % (k, labels[k])])
  return result
