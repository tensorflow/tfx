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

"""Constant values for ModelValidator."""

# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'
# Key for model in executor input_dict.
MODEL_KEY = 'model'

# Key for model blessing in executor output_dict.
BLESSING_KEY = 'blessing'

# Keys for artifact (custom) properties.
ARTIFACT_PROPERTY_BLESSED_KEY = 'blessed'
ARTIFACT_PROPERTY_CURRENT_MODEL_URI_KEY = 'current_model'
ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY = 'current_model_id'
ARTIFACT_PROPERTY_BLESSED_MODEL_URI_KEY = 'blessed_model'
ARTIFACT_PROPERTY_BLESSED_MODEL_ID_KEY = 'blessed_model_id'

# Paths to store model eval results for validation.
CURRENT_MODEL_EVAL_RESULT_PATH = 'eval_results/current_model'
BLESSED_MODEL_EVAL_RESULT_PATH = 'eval_results/blessed_model'

# Values for blessing results.
BLESSED_VALUE = 1
NOT_BLESSED_VALUE = 0

# File names for blessing results.
BLESSED_FILE_NAME = 'BLESSED'
NOT_BLESSED_FILE_NAME = 'NOT_BLESSED'
