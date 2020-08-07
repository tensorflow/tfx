# (generated with --quick)

from typing import Any, Set

dataclasses: module
tf: module

class ExecutionSpec:
    __doc__: str
    input_names: Set[str]
    is_remote_op: bool
    output_names: Set[str]
    subgraph: Any
    def __init__(self, subgraph, input_names: Set[str], output_names: Set[str], is_remote_op: bool) -> None: ...
