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
"""Module for DependencyModule."""

import inspect
from typing import Any, Callable, Optional, TypeVar

from tfx.utils import pure_typing_utils
from tfx.utils.di import errors
from tfx.utils.di import providers

_T = TypeVar('_T')


class DependencyModule:
  """Module that holds a set of dependency providers."""

  def __init__(self):
    self._providers: list[providers.Provider] = []

  def add_provider(self, provider: providers.Provider):
    self._providers.append(provider)

  def provide_value(
      self,
      value: Any,
      *,
      name: Optional[str] = None,
  ) -> None:
    self.add_provider(providers.ValueProvider(value, name=name))

  def provide_class(
      self,
      cls: type[Any],
      *,
      singleton: bool = False,
  ) -> None:
    provider = providers.ClassProvider(cls)
    if singleton:
      provider = providers.SingletonProvider(provider)
    self.add_provider(provider)

  def provide_named_class(
      self, name: str, cls: type[Any], *, singleton: bool = False
  ) -> None:
    provider = providers.NamedClassProvider(name, cls)
    if singleton:
      provider = providers.SingletonProvider(provider)
    self.add_provider(provider)

  def call(self, f: Callable[..., _T]) -> _T:
    """Call the given function whose arguments are auto injected."""
    sig = inspect.signature(f)
    kwargs = {}
    for name, param in sig.parameters.items():
      if param.annotation is inspect.Parameter.empty:
        type_hint = None
      else:
        type_hint = param.annotation
      try:
        kwargs[name] = self.get(name, type_hint)
      except errors.NotProvidedError:
        if param.default is inspect.Parameter.empty:
          raise
    return f(**kwargs)

  def get(self, name: str, type_hint: Any):
    for provider in self._providers:
      if provider.match(name, type_hint):
        try:
          return self.call(provider.make_factory(name, type_hint))
        except errors.FalseMatchError:
          continue
    is_optional, _ = pure_typing_utils.maybe_unwrap_optional(type_hint)
    if is_optional:
      return None
    raise errors.NotProvidedError(
        f'No matching providers found for name={name}, type_hint={type_hint}.'
        f' Available providers: {self._providers}'
    )
