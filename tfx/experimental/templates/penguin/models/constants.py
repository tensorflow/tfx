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
"""Constants for the penguin model.

These values can be tweaked to affect model training performance.
"""

# Defines constants for the model. These constants can be determined via
# experiments using TFX Tuner component.
HIDDEN_LAYER_UNITS = 8
OUTPUT_LAYER_UNITS = 3
NUM_LAYERS = 2
LEARNING_RATE = 1e-2

TRAIN_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 10
