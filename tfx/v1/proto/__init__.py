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
"""TFX proto module."""

from tfx.proto import example_gen_pb2

CustomConfig = example_gen_pb2.CustomConfig
Input = example_gen_pb2.Input
Output = example_gen_pb2.Output
SplitConfig = example_gen_pb2.SplitConfig
PayloadFormat = example_gen_pb2.PayloadFormat
del example_gen_pb2

CustomConfig.__doc__ = """
Optional specified configuration for ExampleGen.
"""

Input.__doc__ = """
Specification of the input of ExampleGen.
"""

Output.__doc__ = """
Specification of the output of the ExampleGen.
"""

SplitConfig.__doc__ = """
A config to partition examples into split in ExampleGen.
"""
