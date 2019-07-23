# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Handler for Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tfx.tools.cli.handler import base_handler


class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  # TODO(b/132286477): Update comments after updating methods.

  def create_pipeline(self) -> None:
    """Creates pipeline in Kubeflow."""
    pass

  def update_pipeline(self) -> None:
    """Updates pipeline in Kubeflow."""
    pass

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    pass

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""
    pass

  def compile_pipeline(self) -> None:
    """Compiles pipeline in Kubeflow."""
    pass

  def create_run(self) -> None:
    """Runs a pipeline in Kubeflow."""
    pass

  def delete_run(self) -> None:
    """Deletes a run."""
    pass

  def terminate_run(self) -> None:
    """Stops a run."""
    pass

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pass

  def get_run(self) -> None:
    """Checks run status."""
    pass
