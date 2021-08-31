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
"""Tests for tfx.extensions.google_cloud_big_query.utils."""

import tensorflow as tf
from tfx.extensions.google_cloud_big_query import utils
from google.protobuf import text_format

_EXAMPLE_1 = text_format.Parse(
    """
    features {
      feature {
          key: "id"
          value {
            int64_list {
              value: 1
            }
          }
      }
      feature {
        key: "name"
        value {
          bytes_list {
            value: "James"
          }
        }
      }
      feature {
        key: "can_program"
        value {
          int64_list {
            value: 1
          }
        }
      }
      feature {
        key: "income"
        value {
          float_list {
            value: 100.19999694824219
          }
        }
      }
      feature {
        key: "income_history"
        value {
          float_list {
            value: 20.5
            value: 30.5
            value: 100.19999694824219
          }
        }
      }
    }
    """, tf.train.Example())


class UtilsTest(tf.test.TestCase):

  def testRowToExample(self):
    field_to_type = {
        'id': 'INTEGER',
        'name': 'STRING',
        'can_program': 'BOOLEAN',
        'income': 'FLOAT',
        'income_history': 'FLOAT'
    }
    field_to_data = {
        'id': 1,
        'name': 'James',
        'can_program': True,
        'income': 100.2,
        'income_history': [20.5, 30.5, 100.2]
    }

    example = utils.row_to_example(
        field_to_type=field_to_type, field_name_to_data=field_to_data)

    self.assertEqual(example, _EXAMPLE_1)

  def testRowToExampleWithUnsupportedTypes(self):
    field_to_type = {
        'time': 'TIMESTAMP',
    }
    field_to_data = {
        'time': 1603493800000,
    }

    with self.assertRaises(RuntimeError) as context:
      utils.row_to_example(
          field_to_type=field_to_type, field_name_to_data=field_to_data)

    self.assertIn('BigQuery column type TIMESTAMP is not supported.',
                  str(context.exception))


if __name__ == '__main__':
  tf.test.main()
