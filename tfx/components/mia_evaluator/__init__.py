# Copyright 2022 Google LLC. All Rights Reserved.
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
"""TFX MIA Evaluator top-level links."""

# Allow proto types to be imported at the top-level since protos live in
# the mia_evaluator namespace.
# pylint: disable=g-importing-member
from tfx.components.mia_evaluator.mia_config_pb2 import ExamplePaths
from tfx.components.mia_evaluator.mia_config_pb2 import MIACustomConfig
from tfx.components.mia_evaluator.mia_evaluator import MiaEvaluator
from tfx.components.mia_evaluator.mia_evaluator import SPLIT_KEY
from tfx.components.mia_evaluator.mia_evaluator import TEST_SPLIT_NAME
from tfx.components.mia_evaluator.mia_evaluator import TRAIN_SPLIT_NAME
