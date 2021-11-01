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
"""Base class for TFX Beam components."""

from typing import Iterable, cast

from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec


class BaseBeamComponent(base_component.BaseComponent):
  """Base class for a TFX Beam pipeline component.

  An instance of a subclass of BaseBaseComponent represents the parameters for a
  single execution of that TFX Beam pipeline component.

  Beam based components should subclass BaseBeamComponent instead of
  BaseComponent in order to inherit Beam related SDKs. All subclasses of
  BaseBeamComponent should override the required class level attributes
  specified in BaseComponent.
  """

  def with_beam_pipeline_args(
      self, beam_pipeline_args: Iterable[str]) -> 'BaseBeamComponent':
    """Add per component Beam pipeline args.

    Args:
      beam_pipeline_args: List of Beam pipeline args to be added to the Beam
        executor spec.

    Returns:
      the same component itself.
    """
    cast(executor_spec.BeamExecutorSpec,
         self.executor_spec).add_beam_pipeline_args(beam_pipeline_args)
    return self

  @classmethod
  def _validate_component_class(cls):
    """Validate that the SPEC_CLASSES property of this class is set properly."""
    super()._validate_component_class()
    if not isinstance(cls.EXECUTOR_SPEC, executor_spec.BeamExecutorSpec):
      raise TypeError(
          ('Beam component class %s expects EXECUTOR_SPEC property to be an '
           'instance of BeamExecutorSpec; got %s instead.') %
          (cls, type(cls.EXECUTOR_SPEC)))
