import glob
import os

import tensorflow as tf
import tensorflow_data_validation as tfdv
print('TF version: {}'.format(tf.version.VERSION))
print('TFDV version: {}'.format(tfdv.version.__version__))
# Read artifact information from metadata store.
# import beam_dag_runner

from tfx.orchestration import metadata
from tfx.types import standard_artifacts

def helper(meta_path):
	metadata_connection_config = metadata.sqlite_metadata_connection_config(meta_path)
	with metadata.Metadata(metadata_connection_config) as store:
	    stats_artifacts = store.get_artifacts_by_type("Examples")
	    # schema_artifacts = store.get_artifacts_by_type(standard_artifacts.Schema.TYPE_NAME)
	    # anomalies_artifacts = store.get_artifacts_by_type(standard_artifacts.ExampleAnomalies.TYPE_NAME)

	print("stats_artifacts", type(stats_artifacts))
	print("stats_artifacts", len(stats_artifacts))
	print("stats_artifacts", type(stats_artifacts[0]))
	print("stats_artifacts", stats_artifacts)
	for art in stats_artifacts:
		print(art.custom_properties['producer_component'].string_value)
	# print("schema_artifacts", schema_artifacts)
	# print("anomalies_artifacts", anomalies_artifacts)
	# # configure output paths
	# # Exact paths to output artifacts can also be found on KFP Web UI if you are using kubeflow.
	# stats_path = stats_artifacts[-1].uri
	# train_stats_file = os.path.join(stats_path, 'train', 'stats_tfrecord')
	# eval_stats_file = os.path.join(stats_path, 'eval', 'stats_tfrecord')
	# print("Train stats file:{}, Eval stats file:{}".format(
	#     train_stats_file, eval_stats_file))

	# schema_file = os.path.join(schema_artifacts[-1].uri, 'schema.pbtxt')
	# print("Generated schame file:{}".format(schema_file))
	# anomalies_file = os.path.join(anomalies_artifacts[-1].uri, 'anomalies.pbtxt')
	# print("Generated anomalies file:{}".format(anomalies_file))


	# train_stats = tfdv.load_statistics(train_stats_file)
	# eval_stats = tfdv.load_statistics(eval_stats_file)
	# tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
	#                           lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')
	# # load generated schema from SchemaGen
	# schema = tfdv.load_schema_text(schema_file)
	# tfdv.display_schema(schema=schema)
	# # load data vaildation result from ExampleValidator
	# anomalies = tfdv.load_anomalies_text(anomalies_file)
	# tfdv.display_anomalies(anomalies)
def first():
	print("first")
	path = os.path.join(os.environ['HOME'],  'tfx/metadata/chicago_taxi_beam/meta3.db')
	print("path", path)
	helper(path)

def second():
	pass
	# print("second")
	# path2 = os.path.join(os.environ['HOME'],  'tfx/metadata/chicago_taxi_beam/meta2.db')

	# # path2 ='/Users/sujipark/Desktop/metadata/chicago_taxi_beam/metadata.db'# os.path.join(os.environ['HOME'], 'Desktop', 'metadata/chicago_taxi_beam/metadata.db')
	# print("path", path2)
	# helper(path2)

if __name__ == '__main__':
	first()
	second()
'''
# 	print("stats_artifacts", stats_artifacts)
# 	print("schema_artifacts", schema_artifacts)
# 	print("anomalies_artifacts", anomalies_artifacts)
# 	# configure output paths
# 	# Exact paths to output artifacts can also be found on KFP Web UI if you are using kubeflow.
# 	stats_path = stats_artifacts[-1].uri
# 	train_stats_file = os.path.join(stats_path, 'train', 'stats_tfrecord')
# 	eval_stats_file = os.path.join(stats_path, 'eval', 'stats_tfrecord')
# 	print("Train stats file:{}, Eval stats file:{}".format(
# 	    train_stats_file, eval_stats_file))

# 	schema_file = os.path.join(schema_artifacts[-1].uri, 'schema.pbtxt')
# 	print("Generated schame file:{}".format(schema_file))
# 	anomalies_file = os.path.join(anomalies_artifacts[-1].uri, 'anomalies.pbtxt')
# 	print("Generated anomalies file:{}".format(anomalies_file))


# 	train_stats = tfdv.load_statistics(train_stats_file)
# 	eval_stats = tfdv.load_statistics(eval_stats_file)
# 	tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
# 	                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')
# 	# load generated schema from SchemaGen
# 	schema = tfdv.load_schema_text(schema_file)
# 	tfdv.display_schema(schema=schema)
# 	# load data vaildation result from ExampleValidator
# 	anomalies = tfdv.load_anomalies_text(anomalies_file)
# 	tfdv.display_anomalies(anomalies)


path1 = os.path.join(os.environ['HOME'],  'tfx/metadata/chicago_taxi_beam/metadata.db')
path2 = path2 = os.path.join(os.environ['HOME'],  'tfx/metadata/chicago_taxi_beam/meta2.db')

metadata_connection_config1 = metadata.sqlite_metadata_connection_config(path1)
with metadata.Metadata(metadata_connection_config1) as store:
    stats_artifacts1 = store.get_artifacts_by_type(standard_artifacts.ExampleStatistics.TYPE_NAME)
    schema_artifacts1 = store.get_artifacts_by_type(standard_artifacts.Schema.TYPE_NAME)
    anomalies_artifacts1 = store.get_artifacts_by_type(standard_artifacts.ExampleAnomalies.TYPE_NAME)

metadata_connection_config2 = metadata.sqlite_metadata_connection_config(path2)
with metadata.Metadata(metadata_connection_config2) as store:
    stats_artifacts2 = store.get_artifacts_by_type(standard_artifacts.ExampleStatistics.TYPE_NAME)
    schema_artifacts2 = store.get_artifacts_by_type(standard_artifacts.Schema.TYPE_NAME)
    anomalies_artifacts2 = store.get_artifacts_by_type(standard_artifacts.ExampleAnomalies.TYPE_NAME)

# print("stats_artifacts", stats_artifacts1,"\n", stats_artifacts2)
  

# assert stats_artifacts1 == stats_artifacts2
# assert schema_artifacts1 == schema_artifacts2
# assert anomalies_artifacts1 == anomalies_artifacts2
# print("stats_artifacts", stats_artifacts)
# print("schema_artifacts", schema_artifacts)
# print("anomalies_artifacts", anomalies_artifacts)
# # configure output paths
# # Exact paths to output artifacts can also be found on KFP Web UI if you are using kubeflow.
stats_path = stats_artifacts[-1].uri
train_stats_file = os.path.join(stats_path, 'train', 'stats_tfrecord')
eval_stats_file = os.path.join(stats_path, 'eval', 'stats_tfrecord')
print("Train stats file:{}, Eval stats file:{}".format(
    train_stats_file, eval_stats_file))

schema_file = os.path.join(schema_artifacts[-1].uri, 'schema.pbtxt')
print("Generated schame file:{}".format(schema_file))
anomalies_file = os.path.join(anomalies_artifacts[-1].uri, 'anomalies.pbtxt')
print("Generated anomalies file:{}".format(anomalies_file))


train_stats = tfdv.load_statistics(train_stats_file)
eval_stats = tfdv.load_statistics(eval_stats_file)
tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')
# load generated schema from SchemaGen
schema = tfdv.load_schema_text(schema_file)
tfdv.display_schema(schema=schema)
# load data vaildation result from ExampleValidator
anomalies = tfdv.load_anomalies_text(anomalies_file)
tfdv.display_anomalies(anomalies)'''