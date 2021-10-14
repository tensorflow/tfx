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
"""Tests for AI Platform Training component."""

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tfx.dsl.component.experimental import placeholders
from tfx.orchestration.kubeflow.v2.components.experimental import ai_platform_training_component
from tfx.orchestration.kubeflow.v2.components.experimental import ai_platform_training_executor
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils


class AiPlatformTrainingComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    examples_artifact = standard_artifacts.Examples()
    examples_artifact.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self.examples = channel_utils.as_channel([examples_artifact])

  def testConstructSingleVmJob(self):
    training = ai_platform_training_component.create_ai_platform_training(
        name='my_training_step',
        project_id='my-project',
        region='us-central1',
        image_uri='gcr.io/my-project/caip-training-test:latest',
        args=[
            '--examples',
            placeholders.InputUriPlaceholder('examples'), '--n-steps',
            placeholders.InputValuePlaceholder('n_step'), '--model-dir',
            placeholders.OutputUriPlaceholder('model')
        ],
        scale_tier='BASIC_GPU',
        inputs={
            'examples': self.examples,
        },
        outputs={'model': standard_artifacts.Model},
        parameters={'n_step': 100})

    expected_aip_config = {
        ai_platform_training_executor.PROJECT_CONFIG_KEY: 'my-project',
        ai_platform_training_executor.TRAINING_JOB_CONFIG_KEY: {
            'training_input': {
                'scaleTier':
                    'BASIC_GPU',
                'region':
                    'us-central1',
                'masterConfig': {
                    'imageUri': 'gcr.io/my-project/caip-training-test:latest'
                },
                'args': [
                    '--examples',
                    placeholders.InputUriPlaceholder('examples'), '--n-steps',
                    placeholders.InputValuePlaceholder('n_step'), '--model-dir',
                    placeholders.OutputUriPlaceholder('model')
                ]
            },
            ai_platform_training_executor.LABELS_CONFIG_KEY: None,
        },
        ai_platform_training_executor.JOB_ID_CONFIG_KEY: None,
        ai_platform_training_executor.LABELS_CONFIG_KEY: None,
    }

    # exec_properties has two entries: one is the user-defined 'n_step', another
    # is the aip_training_config.
    self.assertLen(training.exec_properties, 2)
    self.assertEqual(training.outputs['model'].type_name,
                     standard_artifacts.Model.TYPE_NAME)
    self.assertEqual(training.inputs['examples'].type_name,
                     standard_artifacts.Examples.TYPE_NAME)
    self.assertEqual(training.exec_properties['n_step'], 100)
    self.assertEqual(
        training.exec_properties[ai_platform_training_executor.CONFIG_KEY],
        json_utils.dumps(expected_aip_config))

  def testConstructFullSpec(self):
    training_input = {
        'scaleTier':
            'BASIC_GPU',
        'region':
            'us-central1',
        'masterConfig': {
            'imageUri': 'gcr.io/my-project/caip-training-test:latest'
        },
        'args': [
            '--examples',
            placeholders.InputUriPlaceholder('examples'), '--n-steps',
            placeholders.InputValuePlaceholder('n_step'), '--model-dir',
            placeholders.OutputUriPlaceholder('model')
        ]
    }
    training = ai_platform_training_component.create_ai_platform_training(
        name='my_training_step',
        project_id='my-project',
        training_input=training_input,
        inputs={
            'examples': self.examples,
        },
        outputs={'model': standard_artifacts.Model},
        parameters={'n_step': 100})

    expected_aip_config = {
        ai_platform_training_executor.PROJECT_CONFIG_KEY: 'my-project',
        ai_platform_training_executor.TRAINING_JOB_CONFIG_KEY: {
            'training_input': training_input,
            ai_platform_training_executor.LABELS_CONFIG_KEY: None,
        },
        ai_platform_training_executor.JOB_ID_CONFIG_KEY: None,
        ai_platform_training_executor.LABELS_CONFIG_KEY: None,
    }

    self.assertEqual(
        training.exec_properties[
            ai_platform_training_executor.CONFIG_KEY],
        json_utils.dumps(expected_aip_config))

  def testImageUriValidation(self):
    training_input = {
        'scaleTier':
            'BASIC_GPU',
        'region':
            'us-central1',
    }
    with self.assertRaisesRegex(ValueError, 'image_uri is required'):
      _ = ai_platform_training_component.create_ai_platform_training(
          name='my_training_step',
          project_id='my-project',
          training_input=training_input)

  def testRegionValidation(self):
    training_input = {
        'scaleTier':
            'BASIC_GPU',
        'masterConfig': {
            'imageUri': 'gcr.io/my-project/caip-training-test:latest'
        },
        'args': [
            '--my-flag'
        ]
    }
    with self.assertRaisesRegex(ValueError, 'region is required'):
      _ = ai_platform_training_component.create_ai_platform_training(
          name='my_training_step',
          project_id='my-project',
          training_input=training_input)


if __name__ == '__main__':
  tf.test.main()
