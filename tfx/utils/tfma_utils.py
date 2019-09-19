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
"""Utilities for comparing TFMA metrics"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Union

from google.protobuf.wrappers_pb2 import DoubleValue
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto.metrics_for_slice_pb2 import (
    BoundedValue,
    ValueAtCutoffs,
    MetricValue,
)


def double_value_gte(double_value: DoubleValue, threshold: float) -> bool:
    return double_value.value >= threshold


def bounded_value_gte(bounded_value: BoundedValue, threshold: float) -> bool:
    return double_value_gte(bounded_value.lower_bound, threshold)


def value_at_cutoffs_gte(
    value_at_cutoffs: ValueAtCutoffs, cutoff_to_thresholds: Dict[int, float]
) -> bool:
    results = []
    if not isinstance(cutoff_to_thresholds, dict):
        raise TypeError(
            "cutoff_to_thresholds has to be a Dict from cutoff to threshold"
        )
    for value_cutoff_pair in value_at_cutoffs.values:
        if value_at_cutoffs.cutoff in cutoff_to_thresholds:
            results.append(
                double_value_gte(
                    value_cutoff_pair, cutoff_to_thresholds[value_at_cutoffs.cutoff]
                )
            )
    return all(results)


def metric_value_gte(
    metric_value: MetricValue, threshold: Union[float, Dict[int, float]]
) -> bool:
    metric_value_type = metric_value.WhichOneof("type")
    if metric_value_type == "double_value" and isinstance(threshold, float):
        return double_value_gte(metric_value.double_value, threshold)
    elif metric_value_type == "bounded_value" and isinstance(threshold, float):
        return bounded_value_gte(metric_value.bounded_value, threshold)
    elif metric_value_type == "value_at_cutoffs" and isinstance(threshold, dict):
        return value_at_cutoffs_gte(metric_value.value_at_cutoffs, threshold)
    else:
        raise NotImplementedError(
            "metric_value_gte is not able to compare metric values of type {}".format(
                metric_value_type
            )
        )


def compare_double_values(
    left_double_value: DoubleValue, right_double_value: DoubleValue
) -> bool:
    return double_value_gte(left_double_value, right_double_value.value)


def compare_bounded_values(
    left_bounded_value: metrics_for_slice_pb2.BoundedValue,
    right_bounded_value: metrics_for_slice_pb2.BoundedValue,
) -> bool:
    return bounded_value_gte(left_bounded_value, right_bounded_value.lower_bound.value)


def compare_values_at_cutoffs(
    left_value_at_cutoffs: ValueAtCutoffs, right_value_at_cutoffs: ValueAtCutoffs
) -> bool:
    cutoff_to_thresholds = {
        pair.cutoff: pair.value for pair in right_value_at_cutoffs.values
    }
    return value_at_cutoffs_gte(left_value_at_cutoffs, cutoff_to_thresholds)


def compare_metric_values(
    left_metric_value: MetricValue, right_metric_value: MetricValue
) -> bool:
    metric_value_type = left_metric_value.WhichOneof("type")
    right_metric_value_type = right_metric_value.WhichOneof("type")
    if metric_value_type != right_metric_value:
        raise TypeError(
            "Cannot compare {} with {}".format(
                metric_value_type, right_metric_value_type
            )
        )
    if metric_value_type == "double_value":
        return compare_double_values(
            left_metric_value.double_value, right_metric_value.double_value
        )
    elif metric_value_type == "bounded_value":
        return compare_bounded_values(
            left_metric_value.bounded_value, right_metric_value.bounded_value
        )
    elif metric_value_type == "value_at_cutoffs":
        return compare_values_at_cutoffs(
            left_metric_value.value_at_cutoffs, right_metric_value.value_at_cutoffs
        )
    else:
        raise NotImplementedError(
            "compare_metric_values is not able to compare metric values of type {}".format(
                metric_value_type
            )
        )
