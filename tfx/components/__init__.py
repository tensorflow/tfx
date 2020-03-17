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
"""Subpackage for TFX components."""

import tensorflow as tf

# For component user to direct use tfx.components.[...] as an alias.
from tfx.components.bulk_inferrer.component import BulkInferrer
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.common_nodes.resolver_node import ResolverNode
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.big_query_example_gen.component import BigQueryExampleGen
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.infra_validator.component import InfraValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform

# Prevents double logging: TFX and TF uses `tf.logging` but Beam uses standard
# logging, both logging modules add its own handler. Following setting disables
# tf.logging to propagate up to the parent logging handlers. This is a global
# behavior (perhaps thread hostile) which affects all code that uses component
# libaray.
tf.get_logger().propagate = False
