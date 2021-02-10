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
"""Tests for prediction_to_example_utils."""

import tensorflow as tf

from tfx.components.bulk_inferrer import prediction_to_example_utils as utils
from tfx.proto import bulk_inferrer_pb2
from google.protobuf import text_format
from tensorflow_serving.apis import prediction_log_pb2


class PredictionToExampleUtilsTest(tf.test.TestCase):

  def test_convert_for_multi_inference(self):
    prediction_log = text_format.Parse(
        """
      multi_inference_log {
        request {
          input {
            example_list {
              examples {
                features {
                  feature: {
                    key: "input"
                    value: { bytes_list: { value: "feature" } }
                  }
                }
              }
            }
          }
        }
        response {
          results {
            model_spec {
              signature_name: 'classification'
            }
            classification_result {
              classifications {
                classes {
                  label: '1'
                  score: 0.6
                }
                classes {
                  label: '0'
                  score: 0.4
                }
              }
            }
          }
          results {
            model_spec {
              signature_name: 'regression'
            }
            regression_result {
              regressions {
                value: 0.7
              }
            }
          }
        }
      }
    """, prediction_log_pb2.PredictionLog())
    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
          signature_name: 'classification'
          classify_output {
            label_column: 'classify_label'
            score_column: 'classify_score'
          }
        }
        output_columns_spec {
          signature_name: 'regression'
          regress_output {
            value_column: 'regress_value'
          }
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    expected_example = text_format.Parse(
        """
        features {
            feature: {
              key: "input"
              value: { bytes_list: { value: "feature" } }
            }
            feature: {
              key: "classify_label"
              value: { bytes_list: { value: "1" value: "0"} }
            }
            feature: {
              key: "classify_score"
              value: { float_list: { value: 0.6 value: 0.4} }
            }
            feature: {
              key: "regress_value"
              value: { float_list: { value: 0.7} }
            }
          }
    """, tf.train.Example())
    self.assertProtoEquals(expected_example,
                           utils.convert(prediction_log, output_example_spec))

  def test_convert_for_regress(self):
    prediction_log = text_format.Parse(
        """
      regress_log {
        request {
          input {
            example_list {
              examples {
                features {
                  feature: {
                    key: "regress_input"
                    value: { bytes_list: { value: "feature" } }
                  }
                }
              }
            }
          }
        }
        response {
          result {
            regressions {
              value: 0.7
            }
          }
        }
      }
    """, prediction_log_pb2.PredictionLog())

    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
          regress_output {
            value_column: 'regress_value'
          }
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    expected_example = text_format.Parse(
        """
        features {
            feature: {
              key: "regress_input"
              value: { bytes_list: { value: "feature" } }
            }
            feature: {
              key: "regress_value"
              value: { float_list: { value: 0.7 } }
            }
          }
    """, tf.train.Example())
    self.assertProtoEquals(expected_example,
                           utils.convert(prediction_log, output_example_spec))

  def test_convert_for_classify(self):
    prediction_log = text_format.Parse(
        """
      classify_log {
        request {
          input {
            example_list {
              examples {
                features {
                  feature: {
                    key: "classify_input"
                    value: { bytes_list: { value: "feature" } }
                  }
                }
              }
            }
          }
        }
        response {
          result {
            classifications {
              classes {
                label: '1'
                score: 0.6
              }
              classes {
                label: '0'
                score: 0.4
              }
            }
          }
        }
      }
    """, prediction_log_pb2.PredictionLog())
    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
          classify_output {
            label_column: 'classify_label'
            score_column: 'classify_score'
          }
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    expected_example = text_format.Parse(
        """
        features {
            feature: {
              key: "classify_input"
              value: { bytes_list: { value: "feature" } }
            }
            feature: {
              key: "classify_label"
              value: { bytes_list: { value: "1" value: "0"} }
            }
            feature: {
              key: "classify_score"
              value: { float_list: { value: 0.6 value: 0.4} }
            }
          }
    """, tf.train.Example())
    self.assertProtoEquals(expected_example,
                           utils.convert(prediction_log, output_example_spec))

  def test_convert_for_predict(self):
    example = text_format.Parse(
        """
      features {
        feature { key: "predict_input" value: { bytes_list: { value: "feature" } } }
      }""", tf.train.Example())
    prediction_log = text_format.Parse(
        """
      predict_log {
        request {
          inputs {
            key: "%s"
            value {
              dtype: DT_STRING
              tensor_shape { dim { size: 1 } }
            }
          }
       }
       response {
         outputs {
           key: "output_float"
           value {
             dtype: DT_FLOAT
             tensor_shape { dim { size: 1 } dim { size: 2 }}
             float_val: 0.1
             float_val: 0.2
           }
         }
         outputs {
           key: "output_bytes"
           value {
             dtype: DT_STRING
             tensor_shape { dim { size: 1 }}
             string_val: "prediction"
           }
         }
       }
     }
    """ % (utils.INPUT_KEY), prediction_log_pb2.PredictionLog())

    # The ending quote cannot be recognized correctly when `string_val` field
    # is directly set with a serialized string quoted in the text format.
    prediction_log.predict_log.request.inputs[
        utils.INPUT_KEY].string_val.append(example.SerializeToString())

    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
          predict_output {
            output_columns {
              output_key: 'output_float'
              output_column: 'predict_float'
            }
            output_columns {
              output_key: 'output_bytes'
              output_column: 'predict_bytes'
            }
          }
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    expected_example = text_format.Parse(
        """
        features {
            feature: {
              key: "predict_input"
              value: { bytes_list: { value: "feature" } }
            }
            feature: {
              key: "predict_float"
              value: { float_list: { value: 0.1 value: 0.2 } }
            }
            feature: {
              key: "predict_bytes"
              value: { bytes_list: { value: "prediction" } }
            }
          }
    """, tf.train.Example())
    self.assertProtoEquals(expected_example,
                           utils.convert(prediction_log, output_example_spec))

  def test_convert_for_regress_invalid_output_example_spec(self):
    prediction_log = text_format.Parse(
        """
      regress_log {
        request {
          input {
            example_list {
              examples {
                features {
                  feature: {
                    key: "regress_input"
                    value: { bytes_list: { value: "feature" } }
                  }
                }
              }
            }
          }
        }
        response {
          result {
            regressions {
              value: 0.7
            }
          }
        }
      }
    """, prediction_log_pb2.PredictionLog())

    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    with self.assertRaises(ValueError):
      utils.convert(prediction_log, output_example_spec)

  def test_convert_for_classify_invalid_output_example_spec(self):
    prediction_log = text_format.Parse(
        """
      classify_log {
        request {
          input {
            example_list {
              examples {
                features {
                  feature: {
                    key: "classify_input"
                    value: { bytes_list: { value: "feature" } }
                  }
                }
              }
            }
          }
        }
        response {
          result {
            classifications {
              classes {
                label: '1'
                score: 0.6
              }
              classes {
                label: '0'
                score: 0.4
              }
            }
          }
        }
      }
    """, prediction_log_pb2.PredictionLog())
    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    with self.assertRaises(ValueError):
      utils.convert(prediction_log, output_example_spec)

  def test_convert_for_predict_invalid_output_example_spec(self):
    example = text_format.Parse(
        """
      features {
        feature { key: "predict_input" value: { bytes_list: { value: "feature" } } }
      }""", tf.train.Example())
    prediction_log = text_format.Parse(
        """
      predict_log {
        request {
          inputs {
            key: "%s"
            value {
              dtype: DT_STRING
              tensor_shape { dim { size: 1 } }
            }
          }
       }
       response {
         outputs {
           key: "output_float"
           value {
             dtype: DT_FLOAT
             tensor_shape { dim { size: 1 } dim { size: 2 }}
             float_val: 0.1
             float_val: 0.2
           }
         }
         outputs {
           key: "output_bytes"
           value {
             dtype: DT_STRING
             tensor_shape { dim { size: 1 }}
             string_val: "prediction"
           }
         }
       }
     }
    """ % (utils.INPUT_KEY), prediction_log_pb2.PredictionLog())

    # The ending quote cannot be recognized correctly when `string_val` field
    # is directly set with a serialized string quoted in the text format.
    prediction_log.predict_log.request.inputs[
        utils.INPUT_KEY].string_val.append(example.SerializeToString())

    output_example_spec = text_format.Parse(
        """
        output_columns_spec {
        }
    """, bulk_inferrer_pb2.OutputExampleSpec())
    with self.assertRaises(ValueError):
      utils.convert(prediction_log, output_example_spec)


if __name__ == '__main__':
  tf.test.main()
