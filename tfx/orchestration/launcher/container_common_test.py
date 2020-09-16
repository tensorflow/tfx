# Lint as: python2, python3
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
"""Tests for tfx.orchestration.launcher.container_common."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kubernetes import client
import tensorflow as tf

from tfx.components.base import executor_spec
from tfx.orchestration.launcher import container_common
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts


class ContainerUtilsTest(tf.test.TestCase):

  def testResolveContainerTemplate(self):
    container_spec = executor_spec.ExecutorContainerSpec(
        image='gcr.io/my/trainer:{{exec_properties.version}}',
        command=['{{exec_properties.model}}_trainer'],
        args=[
            '--steps',
            '{{exec_properties.train_args.num_steps}}',
            '--examples',
            '{{input_dict["examples"]|join(",",attribute="uri")}}',
            '--model-path',
            '{{output_dict["model"][0].uri}}',
        ])
    examples_artifact_1 = standard_artifacts.Examples()
    examples_artifact_1.uri = 'gcs://examples/1'
    examples_artifact_2 = standard_artifacts.Examples()
    examples_artifact_2.uri = 'gcs://examples/2'
    model = standard_artifacts.Model()
    model.uri = 'gcs://model'
    input_dict = {'examples': [examples_artifact_1, examples_artifact_2]}
    output_dict = {'model': [model]}
    exec_properties = {
        'version': 'v1',
        'model': 'cnn',
        'train_args': trainer_pb2.TrainArgs(num_steps=10000),
    }

    actual_spec = container_common.resolve_container_template(
        container_spec, input_dict, output_dict, exec_properties)

    self.assertEqual('gcr.io/my/trainer:v1', actual_spec.image)
    self.assertListEqual(['cnn_trainer'], actual_spec.command)
    self.assertListEqual([
        '--steps',
        '10000',
        '--examples',
        'gcs://examples/1,gcs://examples/2',
        '--model-path',
        'gcs://model',
    ], actual_spec.args)

  def testToSwaggerDict(self):
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(owner_references=[
            client.V1OwnerReference(
                api_version='argoproj.io/v1alpha1',
                kind='Workflow',
                name='wf-1',
                uid='wf-uid-1')
        ]),
        spec=client.V1PodSpec(containers=[], service_account='sa-1'))

    pod_dict = container_common.to_swagger_dict(pod)

    self.assertDictEqual(
        {
            'metadata': {
                'ownerReferences': [{
                    'apiVersion': 'argoproj.io/v1alpha1',
                    'kind': 'Workflow',
                    'name': 'wf-1',
                    'uid': 'wf-uid-1'
                }]
            },
            'spec': {
                'serviceAccount': 'sa-1'
            }
        }, pod_dict)


if __name__ == '__main__':
  tf.test.main()
