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
"""Allows Placeholders to be compared."""

from tfx.dsl.experimental.conditionals import predicate


class Comparable:
  """Allows Placeholders to be compared.

  When a Placeholder class subclasses this, comparing its instances with
  anything else would produce a Predicate.
  """

  def __eq__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.EQUAL, left=self, right=other)

  def __ne__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.NOT_EQUAL, left=self, right=other)

  def __lt__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.LESS_THAN, left=self, right=other)

  def __le__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.LESS_THAN_OR_EQUAL, left=self, right=other)

  def __gt__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.GREATER_THAN, left=self, right=other)

  def __ge__(self, other: predicate.ValueLikeTypes) -> predicate.Predicate:
    return predicate.Predicate.from_comparison(
        predicate.CompareOp.GREATER_THAN_OR_EQUAL, left=self, right=other)
