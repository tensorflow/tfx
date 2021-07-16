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
"""Tests for tfx.dsl.component.experimental.executor_specs."""

import tensorflow as tf
from tfx import types
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.types import channel
from tfx.types import component_spec
from tfx.types import standard_artifacts


class TestComponentSpec(types.ComponentSpec):
  INPUTS = {
      'input_artifact':
          component_spec.ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      'output_artifact':
          component_spec.ChannelParameter(type=standard_artifacts.Model),
  }
  PARAMETERS = {
      'input_parameter': component_spec.ExecutionParameter(type=int),
  }


class ExecutorSpecsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._text = 'text'
    self._input_value_placeholder = placeholders.InputValuePlaceholder(
        'input_artifact')
    self._another_input_value_placeholder = placeholders.InputValuePlaceholder(
        'input_parameter')
    self._input_uri_placeholder = placeholders.InputUriPlaceholder(
        'input_artifact')
    self._output_uri_placeholder = placeholders.OutputUriPlaceholder(
        'output_artifact')
    self._concat_placeholder = placeholders.ConcatPlaceholder([
        self._text, self._input_value_placeholder, self._input_uri_placeholder,
        self._output_uri_placeholder,
    ])
    self._text_concat_placeholder = placeholders.ConcatPlaceholder(
        [self._text, 'text1', placeholders.ConcatPlaceholder(['text2']),])

  def testEncodeTemplatedExecutorContainerSpec(self):
    specs = executor_specs.TemplatedExecutorContainerSpec(
        image='image',
        command=[
            self._text, self._input_value_placeholder,
            self._another_input_value_placeholder, self._input_uri_placeholder,
            self._output_uri_placeholder, self._concat_placeholder
        ])
    encode_result = specs.encode(
        component_spec=TestComponentSpec(
            input_artifact=channel.Channel(type=standard_artifacts.Examples),
            output_artifact=channel.Channel(type=standard_artifacts.Model),
            input_parameter=42))
    self.assertProtoEquals(
        """
      image: "image"
      commands {
        value {
          string_value: "text"
        }
      }
      commands {
        operator {
          artifact_value_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      key: "input_artifact"
                    }
                  }
                }
              }
            }
          }
        }
      }
      commands {
        placeholder {
          type: EXEC_PROPERTY
          key: "input_parameter"
        }
      }
      commands {
        operator {
          artifact_uri_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      key: "input_artifact"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
      }
      commands {
        operator {
          artifact_uri_op {
            expression {
              operator {
                index_op {
                  expression {
                    placeholder {
                      type: OUTPUT_ARTIFACT
                      key: "output_artifact"
                    }
                  }
                  index: 0
                }
              }
            }
          }
        }
      }
      commands {
        operator {
          concat_op {
            expressions {
              value {
                string_value: "text"
              }
            }
            expressions {
              operator {
                artifact_value_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            key: "input_artifact"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            key: "input_artifact"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
            expressions {
              operator {
                artifact_uri_op {
                  expression {
                    operator {
                      index_op {
                        expression {
                          placeholder {
                            type: OUTPUT_ARTIFACT
                            key: "output_artifact"
                          }
                        }
                        index: 0
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }""", encode_result)

  def testEncodeTemplatedExecutorContainerSpec_withConcatAllText(self):
    specs = executor_specs.TemplatedExecutorContainerSpec(
        image='image',
        command=[
            self._text_concat_placeholder
        ],
        args=[
            self._text_concat_placeholder
        ])
    encode_result = specs.encode()
    self.assertProtoEquals("""
      image: "image"
      commands {
        value {
          string_value: "texttext1text2"
        }
      }
      args {
        value {
          string_value: "texttext1text2"
        }
      }""", encode_result)


if __name__ == '__main__':
  tf.test.main()
