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
"""Factory for instantiating rewriters.

NOTE: For your rewriter to be instanitated, please include it as an import and
a constant for ease of invoking the new rewriter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

from tfx.components.trainer.rewriting import rewriter
from tfx.components.trainer.rewriting import tfjs_rewriter  # pylint: disable=unused-import
from tfx.components.trainer.rewriting import tflite_rewriter  # pylint: disable=unused-import

TFLITE_REWRITER = "TFLiteRewriter"
TFJS_REWRITER = "TFJSRewriter"


def create_rewriter(rewriter_type: Text, *args,
                    **kwargs) -> rewriter.BaseRewriter:
  """Instantiates a new rewriter with the given type and constructor arguments.

  Args:
    rewriter_type: The rewriter subclass to instantiate (can be all lowercase).
    *args: Positional initialization arguments to pass to the rewriter.
    **kwargs: Keyward initialization arguments to pass to the rewriter.

  Returns:
    The instantiated rewriter.
  Raises:
    ValueError: If unable to instantiate the rewriter.
  """
  for c in rewriter.BaseRewriter.__subclasses__():
    if (c.__name__.lower()) == rewriter_type.lower():
      return c(*args, **kwargs)
  raise ValueError("Failed to find rewriter: {}".format(rewriter_type))
