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
"""A task scheduler for Importer system node."""

from typing import cast

from tfx import types
from tfx.dsl.components.common import importer
from tfx.orchestration import data_types_utils
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.utils import status as status_lib


class ImporterTaskScheduler(task_scheduler.TaskScheduler[task_lib.ExecNodeTask]
                           ):
  """A task scheduler for Importer system node."""

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    pipeline_node = self.task.get_node()
    output_key = cast(str, self.task.exec_properties[importer.OUTPUT_KEY_KEY])
    output_spec = pipeline_node.outputs.outputs[output_key]
    properties = data_types_utils.build_parsed_value_dict(
        output_spec.artifact_spec.additional_properties)
    custom_properties = data_types_utils.build_parsed_value_dict(
        output_spec.artifact_spec.additional_custom_properties)

    output_artifacts = importer.generate_output_dict(
        metadata_handler=self.mlmd_handle,
        uri=cast(str, self.task.exec_properties[importer.SOURCE_URI_KEY]),
        properties=properties,
        custom_properties=custom_properties,
        reimport=bool(self.task.exec_properties[importer.REIMPORT_OPTION_KEY]),
        output_artifact_class=types.Artifact(
            output_spec.artifact_spec.type).type,
        mlmd_artifact_type=output_spec.artifact_spec.type,
        output_key=output_key)

    return task_scheduler.TaskSchedulerResult(
        status=status_lib.Status(code=status_lib.Code.OK),
        output=task_scheduler.ImporterNodeOutput(
            output_artifacts=output_artifacts))

  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    pass
