# Orchestrating TFX Pipelines

## Apache Beam

Several TFX components rely on [Beam](beam.md) for distributed data processing.
In addition, TFX can use Apache Beam to orchestrate and execute the pipeline DAG.
Beam orchestrator uses a different [BeamRunner](https://beam.apache.org/documentation/runners/capability-matrix/)
than the one which is used for component data processing. With the default
[DirectRunner](https://beam.apache.org/documentation/runners/direct/) setup
the Beam orchestrator can be used for local debugging without incurring the
extra Airflow or Kubeflow dependencies, which simplifies system configuration.

See the
[TFX example on Beam](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py)
for details.
