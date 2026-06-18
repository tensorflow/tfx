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
"""TFX components module."""

# Components.
try:
  from tfx.components.bulk_inferrer.component import BulkInferrer
except ImportError:
  BulkInferrer = None

try:
  from tfx.components.evaluator.component import Evaluator
except ImportError:
  Evaluator = None

try:
  from tfx.components.example_diff.component import ExampleDiff
except ImportError:
  ExampleDiff = None

try:
  from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
except ImportError:
  CsvExampleGen = None

try:
  from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
except ImportError:
  ImportExampleGen = None

try:
  from tfx.components.example_validator.component import ExampleValidator
except ImportError:
  ExampleValidator = None

try:
  from tfx.components.infra_validator.component import InfraValidator
except ImportError:
  InfraValidator = None

try:
  from tfx.components.pusher.component import Pusher
except ImportError:
  Pusher = None

try:
  from tfx.components.schema_gen.component import SchemaGen
except ImportError:
  SchemaGen = None

try:
  from tfx.components.schema_gen.import_schema_gen.component import ImportSchemaGen
except ImportError:
  ImportSchemaGen = None

try:
  from tfx.components.statistics_gen.component import StatisticsGen
except ImportError:
  StatisticsGen = None

try:
  from tfx.components.trainer.component import Trainer
except ImportError:
  Trainer = None

try:
  from tfx.components.transform.component import Transform
except ImportError:
  Transform = None

try:
  from tfx.components.tuner.component import Tuner
except ImportError:
  Tuner = None

# For UDF needs.
# pylint: disable=g-bad-import-order
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
try:
  from tfx.components.tuner.component import TunerFnResult
except ImportError:
  TunerFnResult = None

# pylint: enable=g-bad-import-order
__all__ = [
    "BulkInferrer",
    "CsvExampleGen",
    "DataAccessor",
    "Evaluator",
    "ExampleDiff",
    "ExampleValidator",
    "FnArgs",
    "ImportExampleGen",
    "ImportSchemaGen",
    "InfraValidator",
    "Pusher",
    "SchemaGen",
    "StatisticsGen",
    "Trainer",
    "Transform",
    "Tuner",
    "TunerFnResult",
]
