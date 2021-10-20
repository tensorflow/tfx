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
"""Definition of TFX runner base class."""

import abc
from typing import Any, Optional

from tfx.dsl.compiler import compiler
from tfx.dsl.components.base import base_component
from tfx.orchestration import pipeline as pipeline_py
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import doc_controls


class TfxRunner(metaclass=abc.ABCMeta):
  """Base runner class for TFX.

  This is the base class for every TFX runner.
  """

  @abc.abstractmethod
  def run(self, pipeline: pipeline_py.Pipeline) -> Optional[Any]:
    """Runs a TFX pipeline on a specific platform.

    Args:
      pipeline: a pipeline.Pipeline instance representing a pipeline definition.

    Returns:
      Optional platform-specific object.
    """
    pass


@doc_controls.do_not_generate_docs
class IrBasedRunner(TfxRunner, metaclass=abc.ABCMeta):
  """Base class for IR-based TFX runners."""

  @doc_controls.do_not_doc_inheritable
  @abc.abstractmethod
  def run_with_ir(self, pipeline: pipeline_pb2.Pipeline) -> Optional[Any]:
    """Runs a TFX pipeline on a specific platform.

    Args:
      pipeline: a pipeline_pb2.Pipeline instance representing a pipeline
        definition.

    Returns:
      Optional platform-specific object.
    """
    pass

  def run(self, pipeline: pipeline_py.Pipeline) -> Optional[Any]:
    """See TfxRunner."""
    if isinstance(pipeline, pipeline_pb2.Pipeline):
      raise ValueError(
          'The "run" method, which is only meant for running Pipeline objects, '
          'was called with a Pipeline IR. Did you mean to call the '
          '"run_with_ir" method instead?')
    for component in pipeline.components:
      # TODO(b/187122662): Pass through pip dependencies as a first-class
      # component flag.
      if isinstance(component, base_component.BaseComponent):
        component._resolve_pip_dependencies(  # pylint: disable=protected-access
            pipeline.pipeline_info.pipeline_root)
    c = compiler.Compiler()
    pipeline_pb = c.compile(pipeline)
    return self.run_with_ir(pipeline_pb)
