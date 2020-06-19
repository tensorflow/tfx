import glob
import os

import tensorflow as tf
from tfx.types.artifact import Artifact
import tensorflow_data_validation as tfdv
from tfx.orchestration import metadata
from distutils.dir_util import copy_tree

print('TF version: {}'.format(tf.version.VERSION))
print('TFDV version: {}'.format(tfdv.version.__version__))
# Read artifact information from metadata store.
# import beam_dag_runner

from tfx.orchestration import metadata
from tfx.types import standard_artifacts

metadata_dir = os.path.join(os.environ['HOME'],  'tfx/tfx/examples/chicago_taxi_pipeline/metadata.db')
# record_dir = os.path.join(os.environ['HOME'], "tfx/tfx/recorded")
record_dir = os.path.join(os.environ['HOME'], 'tfx/tfx/examples/chicago_taxi_pipeline/testdata')

metadata_config = metadata.sqlite_metadata_connection_config(metadata_dir)
with metadata.Metadata(metadata_config) as m:
	for artifact in m.store.get_artifacts():
		if artifact == 'ExternalArtifact':
			continue
		src_path = artifact.uri
		dest_path = src_path.replace(os.path.join(os.environ['HOME'], "tfx/pipelines/chicago_taxi_beam/"), "")
		dest_path = dest_path[:dest_path.rfind('/')] # remove trailing number
		dest_path = os.path.join(record_dir, dest_path)

		os.makedirs(dest_path, exist_ok=True)
		copy_tree(src_path, dest_path)