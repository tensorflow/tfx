# Copyright 2022 Google LLC. All Rights Reserved.
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
"""TFX notebook utils."""

import builtins
import functools

from absl import logging


def requires_ipython(fn):
  """Decorator for methods that can only be run in IPython."""

  @functools.wraps(fn)
  def run_if_ipython(*args, **kwargs):
    """Invokes `fn` if called from IPython, otherwise just emits a warning."""
    if getattr(builtins, '__IPYTHON__', None):
      # __IPYTHON__ variable is set by IPython, see
      # https://ipython.org/ipython-doc/rel-0.10.2/html/interactive/reference.html#embedding-ipython.
      return fn(*args, **kwargs)
    else:
      logging.warning(
          'Method "%s" is a no-op when invoked outside of IPython.',
          fn.__name__)

  return run_if_ipython
