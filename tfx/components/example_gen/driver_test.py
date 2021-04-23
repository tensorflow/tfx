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
"""Tests for tfx.components.example_gen.driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tfx.components.example_gen import driver
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_driver
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration.portable import data_types as portable_data_types
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.types import artifact_utils
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import proto_utils


class DriverTest(tf.test.TestCase):

  def setUp(self):
    super(DriverTest, self).setUp()

    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Mock metadata and create driver.
    self._mock_metadata = tf.compat.v1.test.mock.Mock()
    self._file_based_driver = driver.FileBasedDriver(self._mock_metadata)

  def testResolveExecProperties(self):
    # Create input dir.
    self._input_base_path = os.path.join(self._test_dir, 'input_base')
    fileio.makedirs(self._input_base_path)

    # Create exec proterties.
    self._exec_properties = {
        standard_component_specs.INPUT_BASE_KEY:
            self._input_base_path,
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='s1',
                        pattern='span{SPAN:2}/version{VERSION:2}/split1/*'),
                    example_gen_pb2.Input.Split(
                        name='s2',
                        pattern='span{SPAN:2}/version{VERSION:2}/split2/*')
                ])),
        standard_component_specs.RANGE_CONFIG_KEY:
            None,
    }

    # Test align of span number.
    span1_v1_split1 = os.path.join(self._input_base_path, 'span01', 'version01',
                                   'split1', 'data')
    io_utils.write_string_file(span1_v1_split1, 'testing11')
    span1_v1_split2 = os.path.join(self._input_base_path, 'span01', 'version01',
                                   'split2', 'data')
    io_utils.write_string_file(span1_v1_split2, 'testing12')
    span2_v1_split1 = os.path.join(self._input_base_path, 'span02', 'version01',
                                   'split1', 'data')
    io_utils.write_string_file(span2_v1_split1, 'testing21')

    # Check that error raised when span does not match.
    with self.assertRaisesRegexp(
        ValueError, 'Latest span should be the same for each split'):
      self._file_based_driver.resolve_exec_properties(self._exec_properties,
                                                      None, None)

    span2_v1_split2 = os.path.join(self._input_base_path, 'span02', 'version01',
                                   'split2', 'data')
    io_utils.write_string_file(span2_v1_split2, 'testing22')
    span2_v2_split1 = os.path.join(self._input_base_path, 'span02', 'version02',
                                   'split1', 'data')
    io_utils.write_string_file(span2_v2_split1, 'testing21')

    # Check that error raised when span matches, but version does not match.
    with self.assertRaisesRegexp(
        ValueError, 'Latest version should be the same for each split'):
      self._file_based_driver.resolve_exec_properties(self._exec_properties,
                                                      None, None)

    span2_v2_split2 = os.path.join(self._input_base_path, 'span02', 'version02',
                                   'split2', 'data')
    io_utils.write_string_file(span2_v2_split2, 'testing22')

    # Test if latest span and version selected when span and version aligns
    # for each split.
    self._file_based_driver.resolve_exec_properties(self._exec_properties, None,
                                                    None)
    self.assertEqual(self._exec_properties[utils.SPAN_PROPERTY_NAME], 2)
    self.assertEqual(self._exec_properties[utils.VERSION_PROPERTY_NAME], 2)
    self.assertRegex(
        self._exec_properties[utils.FINGERPRINT_PROPERTY_NAME],
        r'split:s1,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*\nsplit:s2,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*'
    )
    updated_input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        self._exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        updated_input_config)

    # Check if latest span is selected.
    self.assertProtoEquals(
        """
        splits {
          name: "s1"
          pattern: "span02/version02/split1/*"
        }
        splits {
          name: "s2"
          pattern: "span02/version02/split2/*"
        }""", updated_input_config)

    # Test driver behavior using RangeConfig with static range.
    self._exec_properties[
        standard_component_specs.INPUT_CONFIG_KEY] = proto_utils.proto_to_json(
            example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(
                    name='s1',
                    pattern='span{SPAN:2}/version{VERSION:2}/split1/*'),
                example_gen_pb2.Input.Split(
                    name='s2',
                    pattern='span{SPAN:2}/version{VERSION:2}/split2/*'),
            ]))

    self._exec_properties[
        standard_component_specs.RANGE_CONFIG_KEY] = proto_utils.proto_to_json(
            range_config_pb2.RangeConfig(
                static_range=range_config_pb2.StaticRange(
                    start_span_number=1, end_span_number=2)))
    with self.assertRaisesRegexp(ValueError,
                                 'For ExampleGen, start and end span numbers'):
      self._file_based_driver.resolve_exec_properties(self._exec_properties,
                                                      None, None)

    self._exec_properties[
        standard_component_specs.RANGE_CONFIG_KEY] = proto_utils.proto_to_json(
            range_config_pb2.RangeConfig(
                static_range=range_config_pb2.StaticRange(
                    start_span_number=1, end_span_number=1)))
    self._file_based_driver.resolve_exec_properties(self._exec_properties, None,
                                                    None)
    self.assertEqual(self._exec_properties[utils.SPAN_PROPERTY_NAME], 1)
    self.assertEqual(self._exec_properties[utils.VERSION_PROPERTY_NAME], 1)
    self.assertRegex(
        self._exec_properties[utils.FINGERPRINT_PROPERTY_NAME],
        r'split:s1,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*\nsplit:s2,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*'
    )
    updated_input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        self._exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        updated_input_config)
    # Check if correct span inside static range is selected.
    self.assertProtoEquals(
        """
        splits {
          name: "s1"
          pattern: "span01/version01/split1/*"
        }
        splits {
          name: "s2"
          pattern: "span01/version01/split2/*"
        }""", updated_input_config)

  def testPrepareOutputArtifacts(self):
    examples = standard_artifacts.Examples()
    output_dict = {
        standard_component_specs.EXAMPLES_KEY:
            channel_utils.as_channel([examples])
    }
    exec_properties = {
        utils.SPAN_PROPERTY_NAME: 2,
        utils.VERSION_PROPERTY_NAME: 1,
        utils.FINGERPRINT_PROPERTY_NAME: 'fp'
    }

    pipeline_info = data_types.PipelineInfo(
        pipeline_name='name', pipeline_root=self._test_dir, run_id='rid')
    component_info = data_types.ComponentInfo(
        component_type='type', component_id='cid', pipeline_info=pipeline_info)

    input_artifacts = {}
    output_artifacts = self._file_based_driver._prepare_output_artifacts(  # pylint: disable=protected-access
        input_artifacts, output_dict, exec_properties, 1, pipeline_info,
        component_info)
    examples = artifact_utils.get_single_instance(
        output_artifacts[standard_component_specs.EXAMPLES_KEY])
    base_output_dir = os.path.join(self._test_dir, component_info.component_id)
    expected_uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
        base_output_dir, 'examples', 1)

    self.assertEqual(examples.uri, expected_uri)
    self.assertEqual(
        examples.get_string_custom_property(utils.FINGERPRINT_PROPERTY_NAME),
        'fp')
    self.assertEqual(
        examples.get_int_custom_property(utils.SPAN_PROPERTY_NAME), 2)
    self.assertEqual(
        examples.get_int_custom_property(utils.VERSION_PROPERTY_NAME), 1)

  def testDriverRunFn(self):
    # Create input dir.
    self._input_base_path = os.path.join(self._test_dir, 'input_base')
    fileio.makedirs(self._input_base_path)

    # Fake previous outputs
    span1_v1_split1 = os.path.join(self._input_base_path, 'span01', 'split1',
                                   'data')
    io_utils.write_string_file(span1_v1_split1, 'testing11')
    span1_v1_split2 = os.path.join(self._input_base_path, 'span01', 'split2',
                                   'data')
    io_utils.write_string_file(span1_v1_split2, 'testing12')

    ir_driver = driver.FileBasedDriver(self._mock_metadata)
    example = standard_artifacts.Examples()

    # Prepare output_dic
    example.uri = 'my_uri'  # Will verify that this uri is not changed.
    output_dic = {standard_component_specs.EXAMPLES_KEY: [example]}

    # Prepare output_dic exec_proterties.
    exec_properties = {
        standard_component_specs.INPUT_BASE_KEY:
            self._input_base_path,
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='s1', pattern='span{SPAN:2}/split1/*'),
                    example_gen_pb2.Input.Split(
                        name='s2', pattern='span{SPAN:2}/split2/*')
                ])),
    }
    result = ir_driver.run(
        portable_data_types.ExecutionInfo(
            output_dict=output_dic, exec_properties=exec_properties))
    # Assert exec_properties' values
    exec_properties = result.exec_properties
    self.assertEqual(exec_properties[utils.SPAN_PROPERTY_NAME].int_value, 1)
    updated_input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.INPUT_CONFIG_KEY].string_value,
        updated_input_config)
    self.assertProtoEquals(
        """
        splits {
          name: "s1"
          pattern: "span01/split1/*"
        }
        splits {
          name: "s2"
          pattern: "span01/split2/*"
        }""", updated_input_config)
    self.assertRegex(
        exec_properties[utils.FINGERPRINT_PROPERTY_NAME].string_value,
        r'split:s1,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*\nsplit:s2,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*'
    )
    # Assert output_artifacts' values
    self.assertLen(
        result.output_artifacts[
            standard_component_specs.EXAMPLES_KEY].artifacts, 1)
    output_example = result.output_artifacts[
        standard_component_specs.EXAMPLES_KEY].artifacts[0]
    self.assertEqual(output_example.uri, example.uri)
    self.assertEqual(
        output_example.custom_properties[utils.SPAN_PROPERTY_NAME].int_value, 1)
    self.assertRegex(
        output_example.custom_properties[
            utils.FINGERPRINT_PROPERTY_NAME].string_value,
        r'split:s1,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*\nsplit:s2,num_files:1,total_bytes:9,xor_checksum:.*,sum_checksum:.*'
    )

  def testQueryBasedDriver(self):
    # Create exec proterties.
    exec_properties = {
        standard_component_specs.INPUT_CONFIG_KEY:
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='s1',
                        pattern='select * from table where date=@span_yyyymmdd_utc'
                    ),
                    example_gen_pb2.Input.Split(
                        name='s2',
                        pattern='select * from table2 where date=@span_yyyymmdd_utc'
                    )
                ])),
        standard_component_specs.RANGE_CONFIG_KEY:
            proto_utils.proto_to_json(
                range_config_pb2.RangeConfig(
                    static_range=range_config_pb2.StaticRange(
                        start_span_number=2, end_span_number=2))),
    }
    # Prepare output_dict
    example = standard_artifacts.Examples()
    example.uri = 'my_uri'
    output_dict = {standard_component_specs.EXAMPLES_KEY: [example]}

    query_based_driver = driver.QueryBasedDriver(self._mock_metadata)
    result = query_based_driver.run(
        portable_data_types.ExecutionInfo(
            output_dict=output_dict, exec_properties=exec_properties))

    self.assertEqual(exec_properties[utils.SPAN_PROPERTY_NAME], 2)
    self.assertIsNone(exec_properties[utils.VERSION_PROPERTY_NAME])
    self.assertIsNone(exec_properties[utils.FINGERPRINT_PROPERTY_NAME])
    updated_input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        updated_input_config)
    self.assertProtoEquals(
        """
        splits {
          name: "s1"
          pattern: "select * from table where date='19700103'"
        }
        splits {
          name: "s2"
          pattern: "select * from table2 where date='19700103'"
        }""", updated_input_config)
    self.assertLen(
        result.output_artifacts[
            standard_component_specs.EXAMPLES_KEY].artifacts, 1)
    output_example = result.output_artifacts[
        standard_component_specs.EXAMPLES_KEY].artifacts[0]
    self.assertEqual(output_example.uri, example.uri)
    self.assertEqual(
        output_example.custom_properties[utils.SPAN_PROPERTY_NAME].int_value, 2)


if __name__ == '__main__':
  tf.test.main()
