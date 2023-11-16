# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Modules for dependency injection providers."""

import abc
import functools
import inspect
import sys
from typing import Any, Callable, Generic, Optional, TypeVar, Union, get_args, get_origin

from tfx.utils import pure_typing_utils

_T = TypeVar('_T')


class Provider(abc.ABC, Generic[_T]):
  """Abstraction to provide values in dependency chain.

  Dependency injection is done through the factory functions, where the function
  can contain annotated argument which would be injected from the dependency
  module. Thus both the (argument) name and the type annotation matters.

  Provider defines `match` method to tell if the (name, type_hint) can be
  provided from this instance, and `make_factory` method to build a dependency
  injectable factory function.
  """

  @abc.abstractmethod
  def match(self, name: str, type_hint: Any) -> bool:
    """Whether the requested (name, type_hint) can be provided by self."""
    pass

  @abc.abstractmethod
  def make_factory(self, name: str, type_hint: Any) -> Callable[..., _T]:
    """Build factory class for providing values.

    The factory class can contain additional dependency requests with argument
    type annotations. Both argument name and the argument type matters.

    Factory function may raise FalseMatchError to indicate the match was
    actually False, and needs to lookup for other providers.

    Cyclic depedency would result in infinite recursive call thus not allowed.

    Args:
      name: A name of the injection argument.
      type_hint: A type hint for the injection argument.
    """
    pass


class ValueProvider(Provider[_T]):
  """Simple provider for the singleton value."""

  def __init__(self, value: _T, *, name: Optional[str]):
    self._value = value
    self._name = name

  def match(self, name: str, type_hint: Any) -> bool:
    if self._name and name != self._name:
      return False
    if (
        type_hint is not None
        and not pure_typing_utils.is_compatible(  # pytype: disable=not-supported-yet
            self._value, type_hint
        )
    ):
      return False
    return True

  def make_factory(self, name: str, type_hint: Any) -> Callable[..., Any]:
    return lambda: self._value

  def __repr__(self):
    if self._name:
      return f'ValueProvider(name={self._name}, value={self._value})'
    else:
      return f'ValueProvider({self._value})'


def _is_subclass(cls: type[Any], type_hint: Any) -> bool:
  """issubclass that supports Union and Optional correctly."""
  if inspect.isclass(type_hint) or sys.version_info >= (3, 10):
    # issubclass recognizes Optional / Union type in python>=3.10
    return issubclass(cls, type_hint)
  origin = get_origin(type_hint)
  args = get_args(type_hint)
  if origin is Union:
    if any(_is_subclass(cls, arg) for arg in args):
      return True
  return False


class ClassProvider(Provider[_T]):
  """Simple provider for the class.

  This provider ignores the name of the class, and this provider matches only if
  the type_hint is the base class of this type. For example, if you want to bind
  concrete class to the abstract interface, this provider is the right choice.

  `__init__` argument of the class would be dependency injected.
  """

  def __init__(self, cls: type[_T]):
    self._cls = cls

  def match(self, name: str, type_hint: Any) -> bool:
    if type_hint is None:
      return False
    return _is_subclass(self._cls, type_hint)

  def make_factory(self, name: str, type_hint: Any) -> Callable[..., Any]:
    return self._cls

  def __repr__(self):
    return f'ClassProvider({self._cls.__name__})'


class NamedClassProvider(Provider[_T]):
  """Similar to ClassProvider but name-sensitive."""

  def __init__(self, name: str, cls: type[_T]):
    self._name = name
    self._cls = cls

  def match(self, name: str, type_hint: Any) -> bool:
    return name == self._name and (
        type_hint is None or _is_subclass(self._cls, type_hint)
    )

  def make_factory(self, name: str, type_hint: Any) -> Callable[..., Any]:
    return self._cls

  def __repr__(self):
    return f'NamedClassProvider({self._cls.__name__})'


class SingletonProvider(Provider[_T]):
  """Wrapper provider to provide a singleton instance."""

  def __init__(self, provider: Provider[_T]):
    self._wrapped = provider
    self._value = None

  def match(self, name: str, type_hint: Any) -> bool:
    return self._wrapped.match(name, type_hint)

  def make_factory(self, name: str, type_hint: Any) -> Callable[..., Any]:
    factory = self._wrapped.make_factory(name, type_hint)

    @functools.wraps(factory)
    def singleton_factory(*args, **kwargs):
      if self._value is None:
        self._value = factory(*args, **kwargs)
      return self._value

    return singleton_factory

  def __repr__(self):
    return f'SingletonProvider({self._wrapped})'
