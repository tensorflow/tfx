"""Tests for tfx.experimental.metadata_client.client."""
import os

import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx.components import CsvExampleGen
from tfx.experimental.metadata_client import client
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.pipeline import Pipeline
from tfx.utils.dsl_utils import external_input


def _create_pipeline(root_dir: str):
  root_path = os.path.join(root_dir, "tfx")
  csv_content = ["a,b,c,d", "1,s,3,d"]
  csv_folder = os.path.join(root_path, "csv")
  os.makedirs(csv_folder)
  with open(os.path.join(csv_folder, "test.csv"), "w+") as f:
    f.writelines(csv_content)
  example_gen = CsvExampleGen(
    input=external_input(os.path.join(root_path, "csv")))
  connection_config = metadata_store_pb2.ConnectionConfig()
  connection_config.sqlite.filename_uri = os.path.join(root_path, "db.sqlite")
  connection_config.sqlite.connection_mode = 3  # READWRITE_OPENCREATE
  return Pipeline(
    pipeline_root=root_path,
    pipeline_name="client_test",
    metadata_connection_config=connection_config,
    components=[example_gen],
  )


class MetadataClientTestCase(tf.test.TestCase):
  """MetadataClient tests
  """

  def setUp(self):
    pipeline = _create_pipeline(self.get_temp_dir())
    self.connection_config = pipeline.metadata_connection_config
    self.pipeline_name = pipeline.pipeline_info.pipeline_name
    self.pipeline = pipeline
    super().setUp()

  def test_pipeline_exists(self):
    BeamDagRunner().run(self.pipeline)
    p = client.MetadataClient.from_pipeline(self.pipeline)
    self.assertIsInstance(p, client.PipelineContext)

  def test_pipeline_run_exists(self):
    BeamDagRunner().run(self.pipeline)
    runs = client.MetadataClient.from_pipeline(self.pipeline).runs
    self.assertEqual(len(runs), 1)

  def test_get_status(self):
    BeamDagRunner().run(self.pipeline)
    run = (client.MetadataClient(self.connection_config).get_pipeline(
      self.pipeline_name).runs[0])
    self.assertDictEqual(run.get_status(), {"CsvExampleGen": "complete"})

  def test_pipeline_context_attributes(self):
    BeamDagRunner().run(self.pipeline)
    pipeline = client.MetadataClient.from_pipeline(self.pipeline)
    self.assertEqual(pipeline.name, self.pipeline_name)
    self.assertEqual(pipeline.runs[0].pipeline_name, self.pipeline_name)

  def test_get_pipeline_not_run(self):
    pipeline = client.MetadataClient.from_pipeline(self.pipeline)
    self.assertIsNone(pipeline)


if __name__ == '__main__':
  tf.test.main()
