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

"""Constant values for Evaluator."""

# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'
# Key for model in executor input_dict.
MODEL_KEY = 'model'
# Key for baseline model in executor input_dict.
BASELINE_MODEL_KEY = 'baseline_model'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'

# Key for example splits in executor exec_properties dict.
EXAMPLE_SPLITS_KEY = 'example_splits'

# Key for model blessing in executor output_dict.
BLESSING_KEY = 'blessing'

# Keys for artifact (custom) properties.
ARTIFACT_PROPERTY_BLESSED_KEY = 'blessed'
ARTIFACT_PROPERTY_CURRENT_MODEL_URI_KEY = 'current_model'
ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY = 'current_model_id'
ARTIFACT_PROPERTY_BASELINE_MODEL_URI_KEY = 'baseline_model'
ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY = 'baseline_model_id'

# Values for blessing results.
BLESSED_VALUE = 1
NOT_BLESSED_VALUE = 0

# File names for blessing results.
BLESSED_FILE_NAME = 'BLESSED'
NOT_BLESSED_FILE_NAME = 'NOT_BLESSED'

# Key for evaluation results in executor output_dict.
EVALUATION_KEY = 'evaluation'
