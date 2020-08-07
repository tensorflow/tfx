# (generated with --quick)

from typing import Any

beam: module
beam_pipeline: module
create_complex_graph: module
graph_partition: module
os: module
tempfile: module
test_pipeline: module
tf: module
util: module

class BeamPipelineTest(Any):
    __doc__: str
    def test_validate_outputs(self) -> None: ...

def _extract_outputs(element, graph_name_to_outputs, root_graph) -> list: ...
def _import_tensor_name(node_name) -> str: ...
def _run_original_model(root_graph, root_graph_inputs, graph_name_to_filepath, graph_name_to_outputs) -> list: ...
