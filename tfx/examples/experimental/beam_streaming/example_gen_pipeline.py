
import os
import constants
import absl
from typing import List, Text

from tfx.components import CsvExampleGen
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2
from tfx.utils.dsl_utils import external_input

_pipeline_name = 'taxi_streaming_beam'

# TFX artifact location.
_pipeline_root = os.path.join(constants.TFX_ROOT, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(constants.TFX_ROOT, 'metadata', _pipeline_name,
                              'metadata.db')
# Materialized output data.
_data_root = os.path.join(os.getcwd(), constants.OUTPUT_PATH)

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: Text, pipeline_root: Text, data_root: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
    
    examples = external_input(data_root)
    
    input_config = example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='s1', pattern='{YYYY}-{MM}-{DD}/*')
        ])

    example_gen = CsvExampleGen(
        input=examples,
        input_config=input_config
    )
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_pipeline_args)


def main():
    absl.logging.set_verbosity(absl.logging.INFO)

    BeamDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
          metadata_path=_metadata_path,
          beam_pipeline_args=_beam_pipeline_args))

    
if __name__ == '__main__':
    main()
    