# (generated with --quick)

from typing import Any, Dict, Generator, Optional, Set

ExecuteOneGraph: Any
_ExtractRemoteGraphOutput: Any
_LoadRemoteGraphInputs: Any
beam: module
copy: module
execution_spec: module
tf: module

class _ExecuteOneSubgraphLayer(Any):
    __doc__: str
    def process(self, element: Dict[str, Dict[str, Any]], spec: execution_spec.ExecutionSpec, graph_name_extended: str) -> Generator[Dict[str, Dict[str, Any]], None, None]: ...

def _clear_outputs_for_finished_graph(element: Dict[str, Dict[str, Any]], finished_graph: str) -> Dict[str, Dict[str, Any]]: ...
def _copy_tensor(element: Dict[str, Dict[str, Any]], old_graph: str, old_tensor_name: str, new_graph: str, new_tensor_name: str) -> Dict[str, Dict[str, Any]]: ...
def _get_graph_name(graph_name_extended: str, graph_names: Set[str]) -> Optional[str]: ...
def _import_tensor_name(node_name: str) -> str: ...
