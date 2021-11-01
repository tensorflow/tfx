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
"""Container-based pipeline sample."""

from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import placeholders
from tfx.types import standard_artifacts


downloader_component = container_component.create_container_component(
    name='DownloadFromHttp',
    outputs={
        'data': standard_artifacts.ExternalArtifact,
    },
    parameters={
        'url': str,
    },
    # The component code uses gsutil to upload the data to GCS, so the
    # container image needs to have gsutil installed and configured.
    # Fixing b/150670779 by merging cl/294536017 will lift this limitation.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          url="$0"
          output_data_uri="$1"/data  # TODO(b/150515270) Remove when fixed.
          output_data_path=$(mktemp)

          # Running the main code
          wget "$0" -O "$output_data_path" || curl "$0" > "$output_data_path"

          # Getting data out of the container
          gsutil cp "$output_data_path" "$output_data_uri"
        ''',
        placeholders.InputValuePlaceholder('url'),
        placeholders.OutputUriPlaceholder('data'),
    ],
)


grep_component = container_component.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to GCS, so the
    # container image needs to have gsutil installed and configured.
    # Fixing b/150670779 by merging cl/294536017 will lift this limitation.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$0"
          text_uri="$1"/data  # TODO(b/150515270) Remove when fixed.
          text_path=$(mktemp)
          filtered_text_uri="$2"/data  # TODO(b/150515270) Remove when fixed.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        placeholders.InputValuePlaceholder('pattern'),
        placeholders.InputUriPlaceholder('text'),
        placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)


print_component = container_component.create_container_component(
    name='Print',
    inputs={
        'text': standard_artifacts.ExternalArtifact,
    },
    # The component code uses gsutil to upload the data to GCS, so the
    # container image needs to have gsutil installed and configured.
    # Fixing b/150670779 by merging cl/294536017 will lift this limitation.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          text_uri="$0"/data  # TODO(b/150515270) Remove when fixed.
          text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          cat "$text_path"
        ''',
        placeholders.InputUriPlaceholder('text'),
    ],
)


def create_pipeline_component_instances(text_url: str, pattern: str):
  """Creates tasks for the download_grep_print pipeline."""

  downloader_task = downloader_component(url=text_url)
  grep_task = grep_component(
      text=downloader_task.outputs['data'],
      pattern=pattern,
  )
  print_task = print_component(
      text=grep_task.outputs['filtered_text'],
  )

  component_instances = [
      downloader_task,
      grep_task,
      print_task,
  ]

  return component_instances
