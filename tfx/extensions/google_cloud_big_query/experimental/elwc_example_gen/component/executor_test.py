# Lint as: python3
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
"""Tests for tfx.extensions.google_cloud_big_query.elwc_example_gen.component.executor."""

from unittest import mock

import apache_beam as beam
from apache_beam.testing import util
from google.cloud import bigquery
import tensorflow as tf
from tfx.extensions.google_cloud_big_query import utils
from tfx.extensions.google_cloud_big_query.experimental.elwc_example_gen.component import executor
from tfx.extensions.google_cloud_big_query.experimental.elwc_example_gen.proto import elwc_config_pb2
from tfx.proto import example_gen_pb2

from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow_serving.apis import input_pb2

_ELWC_1 = text_format.Parse(
    """
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 1
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
            float_list {
              value: 1.0
            }
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "one"
            }
          }
        }
      }
    }
    context {
      features {
        feature {
          key: "context_feature_1"
          value {
            int64_list {
              value: 1
            }
          }
        }
        feature {
          key: "context_feature_2"
          value {
            int64_list {
              value: 1
            }
          }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

_ELWC_2 = text_format.Parse(
    """
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
            float_list {
              value: 2.0
            }
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "two"
            }
          }
        }
      }
    }
    context {
      features {
        feature {
          key: "context_feature_1"
          value {
            int64_list {
              value: 1
            }
          }
        }
        feature {
          key: "context_feature_2"
          value {
            int64_list {
              value: 2
            }
          }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

_ELWC_3 = text_format.Parse(
    """
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 3
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
            float_list {
              value: 3.0
            }
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "three"
            }
          }
        }
      }
    }
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 4
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
            float_list {
              value: 4.0
            }
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "four"
            }
          }
        }
      }
    }
    context {
      features {
        feature {
          key: "context_feature_1"
          value {
            int64_list {
              value: 2
            }
          }
        }
        feature {
          key: "context_feature_2"
          value {
            int64_list {
              value: 1
            }
          }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

# 'context_feature_2' has missing value.
_ELWC_4 = text_format.Parse(
    """
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 5
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
            float_list {
              value: 5.0
            }
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "five"
            }
          }
        }
      }
    }
    context {
      features {
        feature {
          key: "context_feature_1"
          value {
            int64_list {
              value: 3
            }
          }
        }
        feature {
          key: "context_feature_2"
          value {
          }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())

# 'feature_id_2' and 'context_feature_2' have missing value.
_ELWC_5 = text_format.Parse(
    """
    examples {
      features {
        feature {
          key: "feature_id_1"
          value {
            int64_list {
              value: 5
            }
          }
        }
        feature {
          key: "feature_id_2"
          value {
          }
        }
        feature {
          key: "feature_id_3"
          value {
            bytes_list {
              value: "five"
            }
          }
        }
      }
    }
    context {
      features {
        feature {
          key: "context_feature_1"
          value {
            int64_list {
              value: 4
            }
          }
        }
        feature {
          key: "context_feature_2"
          value {
          }
        }
      }
    }
    """, input_pb2.ExampleListWithContext())


@beam.ptransform_fn
def _MockReadFromBigQuery(pipeline, query):
  del query  # Unused arg
  mock_query_results = [{
      'context_feature_1': 1,
      'context_feature_2': 1,
      'feature_id_1': 1,
      'feature_id_2': 1.0,
      'feature_id_3': 'one'
  }, {
      'context_feature_1': 1,
      'feature_id_3': 'two',
      'feature_id_1': 2,
      'context_feature_2': 2,
      'feature_id_2': 2.0
  }, {
      'context_feature_1': 2,
      'context_feature_2': 1,
      'feature_id_1': 3,
      'feature_id_2': 3.0,
      'feature_id_3': 'three'
  }, {
      'context_feature_1': 2,
      'context_feature_2': 1,
      'feature_id_1': 4,
      'feature_id_2': 4.0,
      'feature_id_3': ['four']
  }, {
      'context_feature_1': 3,
      'context_feature_2': None,
      'feature_id_1': 5,
      'feature_id_2': [5.0],
      'feature_id_3': 'five'
  }, {
      'context_feature_1': 4,
      'context_feature_2': None,
      'feature_id_1': [5],
      'feature_id_2': None,
      'feature_id_3': 'five'
  }]
  return pipeline | beam.Create(mock_query_results)


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField('context_feature_1', 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField('context_feature_2', 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField('feature_id_1', 'INTEGER', mode='NULLABLE'),
        bigquery.SchemaField('feature_id_2', 'FLOAT', mode='NULLABLE'),
        bigquery.SchemaField('feature_id_3', 'STRING', mode='NULLABLE'),
    ]
    super(ExecutorTest, self).setUp()

  @mock.patch.multiple(
      utils,
      ReadFromBigQuery=_MockReadFromBigQuery,
  )
  @mock.patch.object(bigquery, 'Client')
  def testBigQueryToElwc(self, mock_client):
    # Mock query result schema for _BigQueryElwcConverter.
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema
    elwc_config = elwc_config_pb2.ElwcConfig(
        context_feature_fields=['context_feature_1', 'context_feature_2'])
    packed_custom_config = example_gen_pb2.CustomConfig()
    packed_custom_config.custom_config.Pack(elwc_config)
    with beam.Pipeline() as pipeline:
      elwc_examples = (
          pipeline | 'ToElwc' >> executor._BigQueryToElwc(
              exec_properties={
                  '_beam_pipeline_args': [],
                  'custom_config':
                      json_format.MessageToJson(
                          packed_custom_config,
                          preserving_proto_field_name=True)
              },
              split_pattern='SELECT context_feature_1, context_feature_2, '
              'feature_id_1, feature_id_2, feature_id_3 FROM `fake`'))

      expected_elwc_examples = [_ELWC_1, _ELWC_2, _ELWC_3, _ELWC_4, _ELWC_5]
      util.assert_that(elwc_examples, util.equal_to(expected_elwc_examples))


if __name__ == '__main__':
  tf.test.main()
