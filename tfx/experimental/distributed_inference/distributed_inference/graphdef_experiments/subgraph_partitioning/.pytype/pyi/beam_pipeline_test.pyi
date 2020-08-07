# (generated with --quick)

from typing import Any, Dict, List

beam: module
beam_pipeline: module
create_complex_graph: module
graph_partition: module
os: module
parent_graph_to_remote_graph_input_name_mapping: Dict[str, Dict[str, Dict[str, str]]]
root_graph: str
root_graph_inputs: List[Dict[str, Dict[str, int]]]
tempfile: module
test_pipeline: module
tf: module
util: module

class BeamPipelineTest(Any):
    __doc__: str
    def test_results(self) -> None: ...

def _extract_outputs(element, graph_name_to_outputs, root_graph) -> list: ...
def _import_tensor_name(node_name) -> str: ...
def _run_original_model(root_graph, graph_name_to_filepath, graph_name_to_outputs, root_graph_inputs) -> list: ...
