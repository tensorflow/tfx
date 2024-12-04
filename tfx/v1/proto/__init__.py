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

from tfx.proto.bulk_inferrer_pb2 import (
    ClassifyOutput,
    DataSpec,
    ModelSpec,
    OutputColumnsSpec,
    OutputExampleSpec,
    PredictOutput,
    PredictOutputCol,
    RegressOutput,
)
from tfx.proto.distribution_validator_pb2 import (
    DistributionValidatorConfig,
    FeatureComparator,
)
from tfx.proto.evaluator_pb2 import FeatureSlicingSpec, SingleSlicingSpec
from tfx.proto.example_diff_pb2 import (
    ExampleDiffConfig,
    PairedExampleSkew,
)
from tfx.proto.example_gen_pb2 import (
    CustomConfig,
    Input,
    Output,
    PayloadFormat,
    SplitConfig,
)
from tfx.proto.infra_validator_pb2 import (
    EnvVar,
    EnvVarSource,
    KubernetesConfig,
    LocalDockerConfig,
    PodOverrides,
    RequestSpec,
    SecretKeySelector,
    ServingSpec,
    TensorFlowServing,
    TensorFlowServingRequestSpec,
    ValidationSpec,
)
from tfx.proto.pusher_pb2 import PushDestination, Versioning
from tfx.proto.range_config_pb2 import RangeConfig, RollingRange, StaticRange
from tfx.proto.trainer_pb2 import EvalArgs, TrainArgs
from tfx.proto.transform_pb2 import SplitsConfig
from tfx.proto.tuner_pb2 import TuneArgs
from tfx.v1.proto import orchestration

ModelSpec.__doc__ = """
Specifies the signature name to run the inference in `components.BulkInferrer`.
"""

DataSpec.__doc__ = """
Indicates which splits of examples should be processed in`components.BulkInferrer`.
"""

OutputExampleSpec.__doc__ = """
Defines how the inferrence results map to columns in output example in `components.BulkInferrer`.
"""

OutputColumnsSpec.__doc__ = """
The signature_name should exist in `ModelSpec.model_signature_name`.
You can leave it unset if no more than one `ModelSpec.model_signature_name` is
specified in your bulk inferrer.
"""

ClassifyOutput.__doc__ = """
One type of output_type under `proto.OutputColumnsSpec`.
"""

RegressOutput.__doc__ = """
One type of output_type under `proto.OutputColumnsSpec`.
"""

PredictOutput.__doc__ = """
One type of output_type under `proto.OutputColumnsSpec`.
"""

PredictOutputCol.__doc__ = """
Proto type of output_columns under `proto.PredictOutput`.
"""

FeatureSlicingSpec.__doc__ = """
Slices corresponding to data set in `components.Evaluator`.
"""

SingleSlicingSpec.__doc__ = """
Specifies a single directive for choosing features for slicing.
An empty proto means we do not slice on features (i.e. use the entire data set).
"""

CustomConfig.__doc__ = """
Optional specified configuration for ExampleGen components.
"""

Input.__doc__ = """
Specification of the input of the ExampleGen components.
"""

Output.__doc__ = """
Specification of the output of the ExampleGen components.
"""

SplitConfig.__doc__ = """
A config to partition examples into split in `proto.Output` of ExampleGen.
"""

PayloadFormat.__doc__ = """
Enum to indicate payload format ExampleGen produces.
"""

ServingSpec.__doc__ = """
Defines an environment of the validating infrastructure in `components.InfraValidator`.
"""

ValidationSpec.__doc__ = """
Specification for validation criteria and thresholds in `components.InfraValidator`.
"""

TensorFlowServing.__doc__ = """
TensorFlow Serving docker image (tensorflow/serving) for serving binary.
"""

LocalDockerConfig.__doc__ = """
Docker runtime in a local machine. This is useful when you're running pipeline with infra validator component in your your local machine.
You need to install docker in advance.
"""

KubernetesConfig.__doc__ = """
Kubernetes configuration.
Model server will be launched in the same namespace KFP is running on, as well as same service account will be used (unless specified).
Model server will have `ownerReferences` to the infra validator, which delegates the strict cleanup guarantee to the kubernetes cluster.
"""

PodOverrides.__doc__ = """
Flattened collections of overridable variables for Pod and its submessages.
"""

EnvVar.__doc__ = """
EnvVar represents an environment variable present in a Container.
"""

EnvVarSource.__doc__ = """
EnvVarSource represents a source for the value of an EnvVar.
"""

SecretKeySelector.__doc__ = """
SecretKeySelector selects a key of a Secret.
"""

RequestSpec.__doc__ = """
Optional configuration about making requests from examples input in `components.InfraValidator`.
"""

TensorFlowServingRequestSpec.__doc__ = """
Request spec for building TF Serving requests.
"""

PushDestination.__doc__ = """
Defines the destination of pusher in `components.Pusher`.
"""

Versioning.__doc__ = """
Versioning method for the model to be pushed. Note that This is the semantic TFX provides, therefore depending on the platform, some versioning method might not be compatible.
For example TF Serving only accepts an integer version that is monotonically increasing.
"""

PushDestination.Filesystem.__doc__ = """
File system based destination definition.
"""

RangeConfig.__doc__ = """
RangeConfig is an abstract proto which can be used to describe ranges for different entities in TFX Pipeline.
"""

RollingRange.__doc__ = """
Describes a rolling range.

`[most_recent_span - num_spans + 1,  most_recent_span].`
For example, say you want the range to include only the latest span,
the appropriate RollingRange would simply be:
`RollingRange <  num_spans = 1 >`
The range is clipped based on available data.
ote that num_spans is required in `proto.RollingRange`, while others are optional.
"""

StaticRange.__doc__ = """
Describes a static window within the specified span numbers `[start_span_number, end_span_number]`.
Note that both numbers should be specified for `proto.StaticRange`.
"""

TrainArgs.__doc__ = """
Args specific to training in `components.Trainer`.
"""

EvalArgs.__doc__ = """
Args specific to eval in `components.Trainer`.
"""

SplitsConfig.__doc__ = """
Defines the splits config in `components.Transform`.
"""

TuneArgs.__doc__ = """
Args specific to tuning in `components.Tuner`.
"""

ExampleDiffConfig.__doc__ = """
Configurations related to Example Diff.
"""

FeatureComparator.__doc__ = """
Per feature configuration in Distribution Validator.
"""

DistributionValidatorConfig.__doc__ = """
Configurations related to Distribution Validator.
"""

PairedExampleSkew.__doc__ = """
Configurations related to Example Diff on feature pairing level.
"""

__all__ = [
    "orchestration",
    "ClassifyOutput",
    "CustomConfig",
    "DataSpec",
    "DistributionValidatorConfig",
    "EnvVar",
    "EnvVarSource",
    "EvalArgs",
    "ExampleDiffConfig",
    "FeatureComparator",
    "FeatureSlicingSpec",
    "Filesystem",
    "Input",
    "KubernetesConfig",
    "LocalDockerConfig",
    "ModelSpec",
    "Output",
    "OutputColumnsSpec",
    "OutputExampleSpec",
    "PairedExampleSkew",
    "PodOverrides",
    "PredictOutput",
    "PredictOutputCol",
    "PushDestination",
    "RangeConfig",
    "RegressOutput",
    "RequestSpec",
    "RollingRange",
    "SecretKeySelector",
    "ServingSpec",
    "SingleSlicingSpec",
    "SplitConfig",
    "SplitsConfig",
    "StaticRange",
    "TensorFlowServing",
    "TensorFlowServingRequestSpec",
    "TrainArgs",
    "TuneArgs",
    "ValidationSpec",
]
