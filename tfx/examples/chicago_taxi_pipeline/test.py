from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text
from tfx import types
from tfx.types.artifact import Artifact
import absl
import json
import os

from tfx_bsl import tfxio
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio.tf_example_record import TFExampleRecord
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen

read_fn = '/usr/local/google/home/sujip/record/CsvExampleGen.json'
with open(read_fn, "r") as f:
  read_json = json.load(f)
split_uris = []
# print(read_json['input_dict']) # input
for key_name, input_dict in read_json['input_dict'].items():
  for type_name, artifact_json in input_dict.items():
    artifact = Artifact.from_json_dict(artifact_json)
    print(artifact)
    for split in ['train', 'eval']:#artifact_utils.decode_split_names(artifact.split_names):
      uri = os.path.join(artifact.uri, split)
      split_uris.append((split, uri))
for split, uri in split_uris:
  input_uri = io_utils.all_files_pattern(uri)

  examples = tfrecord_input(input_uri)
  example_gen = ImportExampleGen(input=examples)
  print(example_gen)
  '''tfxio_kwargs = {'file_pattern': input_uri}
        # TODO(b/151624179): clean this up after tfx_bsl is released with the
        # below flag.
        if getattr(tfxio, 'TFXIO_HAS_TELEMETRY', False):
          tfxio_kwargs['telemetry_descriptors'] = _TELEMETRY_DESCRIPTORS
        input_tfxio = TFExampleRecord(**tfxio_kwargs)
        print("get_examples_data: input_tfxio.TensorFlowDataset %s", input_tfxio.TensorFlowDataset())
        data = 'TFXIORead[{}]'.format(split) >> input_tfxio.BeamSource()
        print("get_examples_data: data %s", data)
      '''