# (generated with --quick)

from typing import Any, Callable, Set, Type, TypeVar

graph_pb2: module

_T = TypeVar('_T')

class ExecutionSpec:
    __doc__: str
    input_names: Set[str]
    is_remote_op: bool
    output_names: Set[str]
    subgraph: Any
    def __init__(self, subgraph, input_names: Set[str], output_names: Set[str], is_remote_op: bool) -> None: ...

@overload
def dataclass(_cls: Type[_T]) -> Type[_T]: ...
@overload
def dataclass(*, init: bool = ..., repr: bool = ..., eq: bool = ..., order: bool = ..., unsafe_hash: bool = ..., frozen: bool = ...) -> Callable[[Type[_T]], Type[_T]]: ...
