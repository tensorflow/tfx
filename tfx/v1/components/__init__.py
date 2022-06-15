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

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top

# Components.
from tfx.components.bulk_inferrer.component import BulkInferrer
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.infra_validator.component import InfraValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.schema_gen.import_schema_gen.component import ImportSchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.components.tuner.component import Tuner

# For UDF needs.
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult


# For users writing custom components
# Expose BaseComponent only for use as a type annotation.
def _conditional_import():
  import typing
  if typing.TYPE_CHECKING:
    from tfx.dsl.components.base import base_component
    globals()["BaseComponent"] = base_component.BaseComponent


_conditional_import()
del _conditional_import

from tfx.types import artifact_utils
from tfx.components.util import examples_utils
