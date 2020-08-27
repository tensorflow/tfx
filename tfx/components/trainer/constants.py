# Lint as: python2, python3
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
"""Constant values for Trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Key for examples in executor input_dict.
EXAMPLES_KEY = 'examples'
# Key for schema in executor input_dict.
SCHEMA_KEY = 'schema'
# Key for transform graph in executor input_dict.
TRANSFORM_GRAPH_KEY = 'transform_graph'
# Key for base model in executor input_dict.
BASE_MODEL_KEY = 'base_model'
# Key for hyperparameters in executor input_dict.
HYPERPARAMETERS_KEY = 'hyperparameters'

# Key for train args in executor exec_properties.
TRAIN_ARGS_KEY = 'train_args'
# Key for eval args in executor exec_properties.
EVAL_ARGS_KEY = 'eval_args'
# Key for custom config in executor exec_properties.
CUSTOM_CONFIG_KEY = 'custom_config'

# Key for output model in executor output_dict.
MODEL_KEY = 'model'
# Key for log output in executor output_dict
MODEL_RUN_KEY = 'model_run'

# The name of environment variable to indicate distributed training cluster.
TF_CONFIG_ENV = 'TF_CONFIG'
