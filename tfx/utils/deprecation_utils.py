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
"""Internal function and class deprecation utilities.

The functions in this module aim to conform to the API of
`tensorflow.python.util.deprecation` where possible, with noted differences in
behavior.

Internal use only, no backwards compatibility guarantees.
"""
import functools
import inspect
import warnings

_PRINTED_WARNING = set()


def _should_warn(func_or_class, warn_once=True):
  """Check whether to warn or not with side effect."""
  if id(func_or_class) in _PRINTED_WARNING:
    return False
  if warn_once:
    _PRINTED_WARNING.add(id(func_or_class))
  return True


def _validate_callable(func):
  if not hasattr(func, '__call__'):
    raise ValueError('%s passed to is not a function for deprecation.' %
                     (func,))


def _call_location(levels=2):
  """Return caller's location a number of levels up."""
  f = inspect.currentframe()
  if not f:
    return '<unknown location>'
  for _ in range(levels):
    f = f.f_back or f
  return '%s:%s' % (f.f_code.co_filename, f.f_lineno)


def deprecated(date, instructions, warn_once=True):
  """Decorator marking function or method as deprecated.

  Note: this function does not currently support deprecation of classes. To
  perform such deprecation, decorate its constructor instead.

  Args:
    date: String date at which function will be removed, or None.
    instructions: Instructions on updating use of deprecated code.
    warn_once: Whether only one warning should be emitted for multiple calls to
      deprecated symbol.

  Returns:
    Decorated function or method.
  """

  def deprecated_wrapper(func):
    _validate_callable(func)

    @functools.wraps(func)
    def new_func(*args, **kwargs):
      if _should_warn(func, warn_once=warn_once):
        call_loc = _call_location()
        func_name = getattr(func, '__qualname__', func.__name__)
        func_module = func.__module__
        removed_at = f'after {date}' if date else 'in a future version'
        message = (
            f'From {call_loc}: {func_name} (from {func_module}) is deprecated '
            f'and will be removed {removed_at}. Instructions for updating:\n'
            + instructions)
        warn_deprecated(message)
      return func(*args, **kwargs)

    return new_func

  return deprecated_wrapper


def _make_alias_docstring(new_name, func_or_class):
  """Make deprecation alias docstring."""
  if func_or_class.__doc__:
    lines = func_or_class.__doc__.split('\n')
    lines[0] += ' (deprecated)'
  else:
    lines = ['DEPRECATED CLASS']
  first_line = lines[0]
  notice_lines = [
      ('Warning: THIS CLASS IS DEPRECATED. It will be removed in a future '
       'version.'),
      'Please use %s instead.' % new_name
  ]
  remaining_lines = []
  remaining_lines_string = '\n'.join(lines[1:]).strip()
  if remaining_lines_string:
    remaining_lines = remaining_lines_string.split('\n')
  lines = ([first_line, ''] + notice_lines +
           (([''] + remaining_lines) if remaining_lines else []))
  return '\n'.join(lines)


def deprecated_alias(deprecated_name, name, func_or_class, warn_once=True):
  """Deprecates a symbol in favor of a renamed function or class.

  Args:
    deprecated_name: Fully qualified name of deprecated symbol.
    name: New symbol name.
    func_or_class: Non-deprecated function or class, to be used as alias.
    warn_once: Whether only one warning should be emitted for multiple calls to
      deprecated symbol.

  Returns:
    Decorated function or method.
  """
  if inspect.isclass(func_or_class):
    new_doc = _make_alias_docstring(name, func_or_class)

    class _NewDeprecatedClass(func_or_class):  # pylint: disable=empty-docstring
      __doc__ = new_doc
      __name__ = func_or_class.__name__
      __module__ = _call_location(levels=3)

      # Marker so that instrospection can determine that this is a deprecated
      # class wrapper.
      _TFX_DEPRECATED_CLASS = True

      @functools.wraps(func_or_class.__init__)
      def __init__(self, *args, **kwargs):
        _NewDeprecatedClass.__init__.__doc__ = func_or_class.__init__.__doc__
        if _should_warn(_NewDeprecatedClass.__init__, warn_once=warn_once):
          call_loc = _call_location()
          warn_deprecated(
              f'From {call_loc}: The name {deprecated_name} is deprecated. '
              f'Please use {name} instead.')
        super().__init__(*args, **kwargs)

    return _NewDeprecatedClass
  else:
    _validate_callable(func_or_class)

    @functools.wraps(func_or_class)
    def new_func(*args, **kwargs):  # pylint: disable=empty-docstring
      if _should_warn(new_func, warn_once=warn_once):
        call_loc = _call_location()
        warn_deprecated(
            f'From {call_loc}: The name {deprecated_name} is deprecated. '
            f'Please use {name} instead.')
      return func_or_class(*args, **kwargs)

    return new_func


def get_first_nondeprecated_class(cls):
  """Get the first non-deprecated class in class hierarchy.

  For internal use only, no backwards compatibility guarantees.

  Args:
    cls: A class which may be marked as a deprecated alias.

  Returns:
    First class in the given class's class hierarchy (traversed in MRO order)
    that is not a deprecated alias.
  """
  for mro_class in inspect.getmro(cls):
    if mro_class.__name__ != '_NewDeprecatedClass':
      return mro_class


class TfxDeprecationWarning(DeprecationWarning):  # pylint: disable=g-bad-exception-name
  """DeprecationWarning for TFX API."""


def warn_deprecated(msg):
  """Convenient method to warn TfxDeprecationWarning."""
  warnings.warn(msg, TfxDeprecationWarning)
