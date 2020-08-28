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

import absl

from tfx.orchestration import pipeline as pipeline_module
from tfx.orchestration.experimental.kubernetes import kubernetes_dag_runner
from tfx.orchestration.test_pipelines.download_grep_print_pipeline import create_pipeline_component_instances

_pipeline_name = 'download_grep_print_pipeline'

# Directory and data locations (uses Google Cloud Storage).
_pipeline_root = 'gs://my-bucket'

absl.logging.set_verbosity(absl.logging.INFO)


def _create_pipeline() -> pipeline_module.Pipeline:
  """Create sample container component pipeline."""

  pipeline_name = _pipeline_name
  pipeline_root = _pipeline_root

  text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd/data/tinyshakespeare/input.txt'
  pattern = 'art thou'
  components = create_pipeline_component_instances(text_url, pattern)

  # Use the default in-cluster MySql metadata config.
  config = kubernetes_dag_runner.get_default_kubernetes_metadata_config()

  return pipeline_module.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      metadata_connection_config=config,
      enable_cache=False,
  )


def main():
  # First, create the tfx pipeline instance.
  pipeline = _create_pipeline()
  # Use kubernetes dag runner to run the pipeline.
  kubernetes_dag_runner.KubernetesDagRunner().run(pipeline=pipeline)


if __name__ == '__main__':
  main()
