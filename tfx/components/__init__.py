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
# For component user to direct use tfx.components.[...] as an alias.

# Pre-emptively mock tfx_bsl.arrow.sql_util if it is missing (e.g. when ZetaSQL
# was removed) to ensure tensorflow_model_analysis imports fully and correctly.
try:
  import sys
  from unittest import mock
  try:
    import tfx_bsl.arrow.sql_util
  except ImportError:
    mock_sql_util = mock.MagicMock()
    sys.modules['tfx_bsl.arrow.sql_util'] = mock_sql_util

  import tensorflow_model_analysis as _tfma
  from tensorflow_model_analysis.proto import config_pb2 as _config_pb2
  for attr in [
      'EvalConfig', 'ModelSpec', 'SlicingSpec', 'MetricsSpec',
      'MetricConfig', 'MetricThreshold', 'GenericValueThreshold',
      'GenericChangeThreshold', 'MetricDirection'
  ]:
    if hasattr(_config_pb2, attr):
      val = getattr(_config_pb2, attr)
      if not hasattr(_tfma, attr):
        setattr(_tfma, attr, val)
      if hasattr(_tfma, 'sdk') and not hasattr(_tfma.sdk, attr):
        setattr(_tfma.sdk, attr, val)
except Exception:
  pass

try:
  from tfx.components.bulk_inferrer.component import BulkInferrer
except ImportError:
  BulkInferrer = None

try:
  from tfx.components.distribution_validator.component import DistributionValidator
except ImportError:
  DistributionValidator = None

try:
  from tfx.components.evaluator.component import Evaluator
except ImportError:
  Evaluator = None

try:
  from tfx.components.example_diff.component import ExampleDiff
except ImportError:
  ExampleDiff = None

try:
  from tfx.components.example_gen.component import FileBasedExampleGen
except ImportError:
  FileBasedExampleGen = None

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
  from tfx.components.model_validator.component import ModelValidator
except ImportError:
  ModelValidator = None

try:
  from tfx.components.pusher.component import Pusher
except ImportError:
  Pusher = None

try:
  from tfx.components.schema_gen.component import SchemaGen
except ImportError:
  SchemaGen = None

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


__all__ = [
    "BulkInferrer",
    "DistributionValidator",
    "Evaluator",
    "ExampleDiff",
    "FileBasedExampleGen",
    "CsvExampleGen",
    "ImportExampleGen",
    "ExampleValidator",
    "InfraValidator",
    "ModelValidator",
    "Pusher",
    "SchemaGen",
    "StatisticsGen",
    "Trainer",
    "Transform",
    "Tuner",
]
