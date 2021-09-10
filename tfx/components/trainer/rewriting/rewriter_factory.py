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

import importlib

from tfx.components.trainer.rewriting import rewriter

TFLITE_REWRITER = 'TFLiteRewriter'
TFJS_REWRITER = 'TFJSRewriter'


def _load_tflite_rewriter():
  importlib.import_module('tfx.components.trainer.rewriting.tflite_rewriter')


def _load_tfjs_rewriter():
  try:
    importlib.import_module('tensorflowjs')
  except ImportError as e:
    raise RuntimeError(
        'tensorflowjs is not installed. Please install [tfjs] extra '
        'dependencies to use tfjs_rewriter.') from e
  else:
    importlib.import_module('tfx.components.trainer.rewriting.tfjs_rewriter')


class _RewriterFactory:
  """Factory class for rewriters."""
  _LOADERS = {
      TFLITE_REWRITER.lower(): _load_tflite_rewriter,
      TFJS_REWRITER.lower(): _load_tfjs_rewriter,
  }
  _loaded = set()

  @classmethod
  def _maybe_load_public_rewriter(cls, lower_rewriter_type: str):
    if (lower_rewriter_type in cls._LOADERS
        and lower_rewriter_type not in cls._loaded):
      cls._LOADERS[lower_rewriter_type]()
      cls._loaded.add(lower_rewriter_type)

  @classmethod
  def get_rewriter_cls(cls, rewriter_type: str):
    rewriter_type = rewriter_type.lower()
    cls._maybe_load_public_rewriter(rewriter_type)
    for subcls in rewriter.BaseRewriter.__subclasses__():
      if subcls.__name__.lower() == rewriter_type:
        return subcls
    raise ValueError('Failed to find rewriter: {}'.format(rewriter_type))


def create_rewriter(rewriter_type: str, *args,
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
  return _RewriterFactory.get_rewriter_cls(rewriter_type)(*args, **kwargs)
