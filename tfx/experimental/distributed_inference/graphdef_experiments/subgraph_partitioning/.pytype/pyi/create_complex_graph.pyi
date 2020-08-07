# (generated with --quick)

from typing import Any

N: int
NDIMS: int
casted_ids1: Any
casted_ids2: Any
graph_a: Any
graph_b: Any
ids1: Any
ids2: Any
ids_a: Any
ids_b1: Any
ids_b1_preprocessed: Any
ids_b2: Any
left_lower_sum: Any
left_upper_add: Any
left_upper_concat: Any
left_upper_floormod: Any
left_upper_round: Any
left_upper_sum: Any
main_graph: Any
main_result: Any
remote_a0: Any
remote_a1: Any
remote_b0: Any
remote_b1: Any
remote_result_a1: Any
remote_result_a2: Any
result_a: Any
result_b: Any
right_lower_div: Any
right_lower_mul: Any
right_lower_sum: Any
right_upper_add: Any
right_upper_floormod: Any
right_upper_mul: Any
right_upper_round: Any
right_upper_sum: Any
table_a: Any
tf: module

def create_session(graph) -> Any: ...
def remote_op_a(input_ids) -> Any: ...
def remote_op_b(input_ids1, input_ids2) -> Any: ...
def save_examples_as_graphdefs(export_dir) -> None: ...
