# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Decorator for creating ResolverFunction based resolver factory."""
import functools
from typing import Callable

from tfx.dsl.components.common import resolver as resolver_lib
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
import tfx.types


def resolver(
    f: Callable[..., resolver_op.OpNode]
) -> Callable[..., resolver_lib.Resolver]:
  """@resolver decorator for converting resolver function to a resolver factory.

  This is analoguous to the @component decorator which turns a component
  execution function to a component node factory.

  Usage:

      @resolver
      def MyResolver(input_dict):
        return MyOp(input_dict, flag=True)

      my_resolver = MyResolver(
          data=component.outputs['data']
      )

  Args:
    f: A resolver function.

  Returns:
    A factory for the Resolver that can create new Resolver instance with
    channel inputs.
  """
  @functools.wraps(f)
  def wrapper(**channels: tfx.types.Channel) -> resolver_lib.Resolver:
    rf = resolver_function.ResolverFunction(f)
    rf.trace()
    return resolver_lib.Resolver(function=rf, **channels)

  return wrapper
