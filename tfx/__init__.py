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
"""Init module for TFX."""

import os
os.environ['TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK'] = 'true'

# `tfx` is a namespace package.
# https://packaging.python.org/guides/packaging-namespace-packages/#pkgutil-style-namespace-packages
__path__ = __import__('pkgutil').extend_path(__path__, __name__)


# Import version string.
from tfx.version import __version__ as __version__

# Pre-emptively mock tfx_bsl.arrow.sql_util if it is missing (e.g. when ZetaSQL
# was removed) to ensure tensorflow_model_analysis imports fully and correctly.
try:
  import sys
  from unittest import mock
  try:
    import tfx_bsl.arrow.sql_util  # noqa: F401
  except ImportError:
    mock_sql_util = mock.MagicMock()
    sys.modules['tfx_bsl.arrow.sql_util'] = mock_sql_util
except Exception:
  pass
