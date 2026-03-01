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
"""Utilities for version manipulation."""

import re

from absl import logging
from tfx import version


# Version string match patterns. It captures 3 patterns of versions:
# 1. Regular release. For example: 0.24.0;
# 2. RC release. For example: 0.24.0-rc1, which maps to image tag: 0.24.0rc1
# 3. Nightly release. For example, 0.24.0.dev20200910;
#    which maps to an identical image tag: 0.24.0.dev20200910
_REGULAR_NIGHTLY_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+(\.dev\d{8}){0,1}')
_RC_VERSION_PATTERN = re.compile(r'\d+\.\d+\.\d+\-rc\d+')


def get_image_version(version_str: str = version.__version__) -> str:
  """Gets the version for image tag based on SDK version.

  Args:
    version_str: The SDK version.

  Returns:
    Version string representing the image version should be used. For offcially
    released version of TFX SDK, we'll align the SDK and the image versions; For
    'dev' or customized versions we'll use the latest image version.
  """
  if _REGULAR_NIGHTLY_VERSION_PATTERN.fullmatch(version_str):
    # This SDK is a released version.
    return version_str
  elif _RC_VERSION_PATTERN.fullmatch(version_str):
    # For RC versions the hiphen needs to be removed.
    return version_str.replace('-', '')

  logging.info('custom/dev SDK version detected: %s, using latest image '
               'version', version_str)
  return 'latest'
