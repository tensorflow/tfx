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
"""Abstract TFX executor class for Beam powered components."""

from typing import Any, List, Optional, Text

import absl
from tfx.dsl.components.base.base_executor import BaseExecutor

try:
  import apache_beam as beam  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


class BaseBeamExecutor(BaseExecutor):
  """Abstract TFX executor class for Beam powered components."""

  class Context(BaseExecutor.Context):
    """Context class for base Beam excecutor."""

    def __init__(self,
                 beam_pipeline_args: Optional[List[Text]] = None,
                 extra_flags: Optional[List[Text]] = None,
                 tmp_dir: Optional[Text] = None,
                 unique_id: Optional[Text] = None,
                 executor_output_uri: Optional[Text] = None,
                 stateful_working_dir: Optional[Text] = None):
      # TODO(b/174174381): replace beam_pipeline_args with extra_flags after
      # beam_pipeline_args is removed from BaseExecutor
      super().__init__(
          tmp_dir=tmp_dir,
          unique_id=unique_id,
          executor_output_uri=executor_output_uri,
          stateful_working_dir=stateful_working_dir)
      self.beam_pipeline_args = beam_pipeline_args
      self.extra_flags = extra_flags

  # TODO(b/126182711): Look into how to support fusion of multiple executors
  # into same pipeline.
  def _make_beam_pipeline(self) -> _BeamPipeline:
    """Makes beam pipeline."""
    if not beam:
      raise Exception(
          'Apache Beam must be installed to use this functionality.')

    result = beam.Pipeline(argv=self._beam_pipeline_args)

    # TODO(b/159468583): Obivate this code block by moving the warning to Beam.
    #
    # pylint: disable=g-import-not-at-top
    from apache_beam.options.pipeline_options import DirectOptions
    from apache_beam.options.pipeline_options import PipelineOptions
    options = PipelineOptions(self._beam_pipeline_args)
    direct_running_mode = options.view_as(DirectOptions).direct_running_mode
    direct_num_workers = options.view_as(DirectOptions).direct_num_workers
    if direct_running_mode == 'in_memory' and direct_num_workers != 1:
      absl.logging.warning(
          'If direct_num_workers is not equal to 1, direct_running_mode should '
          'be `multi_processing` or `multi_threading` instead of `in_memory` '
          'in order for it to have the desired worker parallelism effect.')

    return result
