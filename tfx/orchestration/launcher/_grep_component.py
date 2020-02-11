# Lint as: python3
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
"""GrepComponent for use in E2E tests."""

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types import standard_artifacts


class GrepComponentSpec(component_spec.ComponentSpec):
  """ComponentSpec for GrepComponent."""

  INPUTS = {
      'input1':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExternalArtifact),
  }
  OUTPUTS = {
      'output1':
          component_spec.ChannelParameter(
              type=standard_artifacts.ExternalArtifact),
  }
  PARAMETERS = {
      'pattern': component_spec.ExecutionParameter(type=str),
  }


class GrepComponent(base_component.BaseComponent):
  """GrepComponent filers input data using a regular expression pattern."""

  SPEC_CLASS = GrepComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorContainerSpec(
      image='alpine',
      command=['sh', '-c', 'grep "$2" <"$0" >"$1"'],
      args=[
          '/tmp/inputs/input1/data',
          '/tmp/outputs/output1/data',
          '{{exec_properties.pattern}}',
      ],
      input_path_uris={
          '/tmp/inputs/input1/data': '{{input_dict["input1"][0].uri}}',
      },
      output_path_uris={
          '/tmp/outputs/output1/data': '{{output_dict["output1"][0].uri}}',
      },
  )

  def __init__(self, input1, pattern, output1=None):
    if not output1:
      output1 = channel_utils.as_channel([
          standard_artifacts.ExternalArtifact(),
      ])

    super(GrepComponent, self).__init__(GrepComponentSpec(
        input1=input1,
        output1=output1,
        pattern=pattern,
    ))


def get_expected_k8s_launcher_pod_spec(
    input1_uri,
    output1_uri,
    pattern,
):
  return {
      'apiVersion': 'v1',
      'kind': 'Pod',
      'metadata': {
          'name': 'pod-name',  # 'test-123-fakecomponent-faketask-123',
          'ownerReferences': [{
              'apiVersion': 'argoproj.io/v1alpha1',
              'kind': 'Workflow',
              'name': 'owner-name',  # 'wf-1',
              'uid': 'owner-uid',  # 'wf-uid-1'
          }]
      },
      'spec': {
          'restartPolicy': 'Never',
          'initContainers': [
              {
                  'name': 'downloader',
                  'image': 'gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
                  'command': [
                      'sh', '-e', '-c',
                      '''\
while (( $# > 0 )); do
  gsutil rsync "$0" "$1"
  shift 2
done
'''
                  ],
                  'args': [
                      input1_uri,
                      '/tmp/inputs/input1/data',
                  ],
                  'volumeMounts': [
                      {
                          'name': 'tmp-inputs-input1-data',
                          'mountPath': '/tmp/inputs/input1',
                      },
                  ],
              },
              {
                  'name': 'main',
                  'image': 'alpine',
                  'command': [
                      'sh', '-c', 'grep "$2" <"$0" >"$1"'
                  ],
                  'args': [
                      '/tmp/inputs/input1/data',
                      '/tmp/outputs/output1/data',
                      pattern,
                  ],
                  'volumeMounts': [
                      {
                          'name': 'tmp-inputs-input1-data',
                          'mountPath': '/tmp/inputs/input1',
                      },
                      {
                          'name': 'tmp-outputs-output1-data',
                          'mountPath': '/tmp/outputs/output1',
                      },
                  ],
              },
              {
                  'name': 'uploader',
                  'image': 'gcr.io/google.com/cloudsdktool/cloud-sdk:latest',
                  'command': [
                      'sh', '-e', '-c',
                      '''\
while (( $# > 0 )); do
  gsutil rsync "$0" "$1"
  shift 2
done
'''
                  ],
                  'args': [
                      '/tmp/outputs/output1/data',
                      output1_uri,
                  ],
                  'volumeMounts': [
                      {
                          'name': 'tmp-outputs-output1-data',
                          'mountPath': '/tmp/outputs/output1',
                      },
                  ],
              },
          ],
          'containers': [{
              'name': 'dummy',
              'image': 'alpine',
              'command': ['true'],
          }],
          'volumes': [
              {'name': 'tmp-inputs-input1-data', 'emptyDir': {}},
              {'name': 'tmp-outputs-output1-data', 'emptyDir': {}},
          ],
          'serviceAccount': 'service-account',  # 'sa-1',
          'serviceAccountName': None
      }
  }
