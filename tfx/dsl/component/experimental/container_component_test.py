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
"""Tests for tfx.dsl.component.experimental.container_component."""

import tensorflow as tf

from tfx.dsl.component.experimental import container_component
from tfx.dsl.placeholder import placeholder as ph
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts


class ContainerComponentTest(tf.test.TestCase):

  def testConstructComponentExample(self):
    container_component.create_container_component(
        name='TrainModel',
        inputs={
            'training_data': simple_artifacts.Dataset,
        },
        outputs={
            'model': standard_artifacts.Model,
        },
        parameters={
            'num_training_steps': int,
        },
        image='gcr.io/my-project/my-trainer',
        command=[
            'python3', 'my_trainer',
            '--training_data_uri', ph.input('training_data').uri,
            '--model_uri', ph.output('model').uri,
            '--num_training-steps',
            ph.exec_property('num_training_steps'),
        ]
    )


if __name__ == '__main__':
  tf.test.main()
