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
"""Tests for tfx.components.example_validator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import tensorflow as tf
import pandas as pd

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import components
from tfx.components.example_validator import executor
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils.dsl_utils import external_input
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import pipeline
from tfx.orchestration import metadata
from google.protobuf import text_format


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self.source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self.eval_stats_artifact = standard_artifacts.ExampleStatistics()
    self.eval_stats_artifact.uri = os.path.join(
        self.source_data_dir, 'statistics_gen')
    self.eval_stats_artifact.split_names = artifact_utils.encode_split_names(
        ['eval'])

    self.schema_artifact = standard_artifacts.Schema()
    self.schema_artifact.uri = os.path.join(self.source_data_dir, 'schema_gen')

    self.validation_output = standard_artifacts.ExampleAnomalies()
    self.validation_output.uri = os.path.join(output_data_dir, 'output')

  def _test_do(self, input_dict, output_dict, exec_properties):
    example_validator_executor = executor.Executor()
    example_validator_executor.Do(input_dict, output_dict, exec_properties)
    self.assertEqual(
        ['anomalies.pbtxt'],
        tf.io.gfile.listdir(self.validation_output.uri))
    anomalies = io_utils.parse_pbtxt_file(
        os.path.join(self.validation_output.uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    self.assertNotEqual(0, len(anomalies.anomaly_info))
    return anomalies

  def testDo(self):
    input_dict = {
        executor.STATISTICS_KEY: [self.eval_stats_artifact],
        executor.SCHEMA_KEY: [self.schema_artifact],
    }
    output_dict = {
        executor.ANOMALIES_KEY: [self.validation_output],
    }

    exec_properties = {}

    self._test_do(input_dict, output_dict, exec_properties)
    # TODO(zhitaoli): Add comparison to expected anomolies.

  def testDoSkewDetection(self):
    # Read in the Chicago Taxi data.csv and intentionally skew it to keep only Cash trips
    original_csv_path = os.path.join(self.source_data_dir,
                                     'external/csv/data.csv')
    skewed_data = pd.read_csv(original_csv_path)
    skewed_data = skewed_data.loc[skewed_data['payment_type'] == 'Cash']

    # Write the skewed df to .csv
    tmp_dir = tempfile.mkdtemp()
    skewed_dir = os.path.join(tmp_dir, 'skewed')
    os.mkdir(skewed_dir)
    skewed_path = os.path.join(skewed_dir, 'skewed_data.csv')
    skewed_data.to_csv(skewed_path)

    # TFX pipeline components to generate a schema and tfrecords file for the skewed data
    examples = external_input(skewed_dir)
    example_gen = CsvExampleGen(input=examples)
    statistics_gen = components.StatisticsGen(
        examples=example_gen.outputs['examples'])
    schema_gen = components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)

    p = pipeline.Pipeline(
        pipeline_name='skewed_chicago_taxi_beam',
        pipeline_root=tmp_dir,
        components=[example_gen, statistics_gen, schema_gen],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            os.path.join(tmp_dir, 'metadata.db')))

    BeamDagRunner().run(p)

    # Construct the skewed statistics artifact from pipeline outputs
    component_dir = os.path.join(tmp_dir, 'StatisticsGen/statistics')
    component_num = os.listdir(component_dir)[0]
    skewed_statistics = standard_artifacts.ExampleStatistics()
    skewed_statistics.uri = os.path.join(component_dir, component_num, 'eval')

    # Set a skew comparator for the 'payment_type' feature
    tmp_schema_pbtxt_dir = os.path.join(tmp_dir, 'SkewedSchemaGen')
    os.mkdir(tmp_schema_pbtxt_dir)
    tmp_schema_pbtxt_path = os.path.join(tmp_schema_pbtxt_dir, 'schema.pbtxt')
    original_schema_pbtxt_path = os.path.join(self.schema_artifact.uri,
                                              'schema.pbtxt')
    shutil.copy2(original_schema_pbtxt_path, tmp_schema_pbtxt_path)

    with open(tmp_schema_pbtxt_path, "r+") as f:
      contents = f.read()
      proto_contents = text_format.Parse(contents, schema_pb2.Schema())

      for feature in proto_contents.feature:
        if feature.name == "payment_type":
          feature.skew_comparator.infinity_norm.threshold = 0.001

      contents = text_format.MessageToString(proto_contents)
      f.seek(0)
      f.write(contents)
      f.truncate()

    self.schema_artifact.uri = tmp_schema_pbtxt_dir

    input_dict = {
        executor.STATISTICS_KEY: [self.eval_stats_artifact],
        executor.SCHEMA_KEY: [self.schema_artifact],
        executor.TRAINING_STATISTICS_KEY: [skewed_statistics],
    }

    output_dict = {
        executor.ANOMALIES_KEY: [self.validation_output],
    }

    exec_properties = {}

    # Generate anomalies and test that tfdv detected the skew
    anomalies = self._test_do(input_dict, output_dict, exec_properties)
    assert 'COMPARATOR_L_INFTY_HIGH' in str(anomalies.anomaly_info)

    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
  tf.test.main()
