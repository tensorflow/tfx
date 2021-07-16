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

import contextlib
import re
import sys
import threading
from typing import Dict, List

from absl import logging
from googleapiclient import http
from tfx import version

# Common label names used.
#
# TODO(b/190444391): Add component label.
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
def scoped_labels(labels: Dict[str, str]):
  """Register thread local labels used in current context."""
  if getattr(_thread_local_labels_state, 'dictionary', None) is None:
    _thread_local_labels_state.dictionary = {}
  for key, value in labels.items():
    _thread_local_labels_state.dictionary[key] = value
  try:
    yield
  finally:
    for key in labels:
      _thread_local_labels_state.dictionary.pop(key)


def _normalize_label(value: str) -> str:
  """Lowercase and replace illegal characters in labels."""
  # See https://cloud.google.com/compute/docs/labeling-resources.
  result = re.sub(r'[^a-z0-9\_\-]', '-', value.lower())
  if len(result) > 63:
    logging.warning('Length of label `%s` exceeds maximum length(63), trimmed.',
                    result)
    return result[:63]
  return result


def make_labels_dict() -> Dict[str, str]:
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

  # Only first-party tfx component's executor telemetry will be collected.
  # All other executors will be recorded as `third_party_executor`.
  if (result.get(LABEL_TFX_EXECUTOR) and
      not result[LABEL_TFX_EXECUTOR].startswith('tfx.')):
    result[LABEL_TFX_EXECUTOR] = 'third_party_executor'

  for k, v in result.items():
    result[k] = _normalize_label(v)
  return result


def make_beam_labels_args() -> List[str]:
  """Make Beam arguments for common labels used in TFX pipelines.

  Returns:
    New Beam pipeline args with labels.
  """
  labels = make_labels_dict()
  # See following file for reference to the '--labels ' flag.
  # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py
  result = []
  for k in sorted(labels):
    result.extend(['--labels', '%s=%s' % (k, labels[k])])
  return result


class TFXHttpRequest(http.HttpRequest):
  """HttpRequest builder that sets a customized useragent header for TFX.

  This is used to track the usage of the TFX on Cloud AI.
  """

  def __init__(self, *args, **kwargs):
    """Construct a HttpRequest.

    Args:
        *args: Positional arguments to pass to the base class constructor.
        **kwargs: Keyword arguments to pass to the base class constructor.
    """
    headers = kwargs.setdefault('headers', {})
    # See Mozilla standard User Agent header Syntax:
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent
    # TODO(b/193915978): Stop relying on '-tfxpipeline-' suffix and use
    # tfx/version instead. More labels set to scoped_labels can also be added
    # the comments variable below, upon needed.
    comments = '(client_context:tfxpipeline;)'
    user_agent = f'tfx/{version.__version__} {comments}'
    headers['user-agent'] = user_agent
    super().__init__(*args, **kwargs)
