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
"""TFX Conditionals."""
from typing import Tuple

import attr
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import context_manager
from tfx.dsl.placeholder import placeholder


@attr.s(auto_attribs=True, kw_only=True)
class ConditionalContext(context_manager.DslContext):
  """DslContext for Cond."""
  predicate: placeholder.Predicate

  def validate(self):
    if any(p.predicate == self.predicate
           for p in self.ancestors
           if isinstance(p, ConditionalContext)):
      raise ValueError(
          f'Nested conditionals with duplicate predicates: {self.predicate}.'
          'Consider merging the nested conditionals.')


def get_predicates(
    node: base_node.BaseNode) -> Tuple[placeholder.Predicate, ...]:
  """Gets all predicates that conditional contexts for the node carry."""
  return tuple(c.predicate
               for c in context_manager.get_contexts(node)
               if isinstance(c, ConditionalContext))


class Cond(context_manager.DslContextManager[None]):
  """Cond context manager that disable containing nodes if predicate is False.

  Cond blocks can be nested to express the nested conditions.

  Usage:

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=EvalConfig(...))

    with Cond(evaluator.outputs['blessing'].future()
              .custom_property('blessed') == 1):
      pusher = Pusher(
          model=trainer.outputs['model'],
          push_destination=PushDestination(...))
  """

  def __init__(self, predicate: placeholder.Predicate):
    self._predicate = predicate

  def create_context(self) -> ConditionalContext:
    return ConditionalContext(predicate=self._predicate)

  def enter(self, context: ConditionalContext) -> None:  # pylint: disable=unused-argument
    return None
