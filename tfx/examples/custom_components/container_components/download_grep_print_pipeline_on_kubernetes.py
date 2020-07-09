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
"""Container-based pipeline on kubernetes sample."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl

from tfx.orchestration import pipeline
from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.orchestration.test_pipelines.download_grep_print_pipeline import create_pipeline_component_instances
from tfx.utils import telemetry_utils

_pipeline_name = 'download_grep_print_pipeline'
_pipeline_root = "gs://tfx-eric-default"
absl.logging.set_verbosity(absl.logging.INFO)


def _create_pipeline() -> pipeline.Pipeline:

    pipeline_name = _pipeline_name
    pipeline_root = _pipeline_root

    components = create_pipeline_component_instances("www.google.com", "google")

    metadata_connection_config = kubernetes_dag_runner.get_default_kubernetes_metadata_config()

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        metadata_connection_config=metadata_connection_config,
        enable_cache=False,
    ) 

if __name__ == '__main__':
    # first, create a tfx pipiline
    _pipeline = _create_pipeline()
    # use kubernetes dag runner to run the pipeline
    kubernetes_dag_runner.KubernetesDagRunner().run(tfx_pipeline=_pipeline)
