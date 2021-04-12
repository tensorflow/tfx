# Version 0.29.0

## Major Features and Improvements

*  Added a simple query based driver that supports Span spec and static_range.
*  Added e2e rolling window example/test for Span Resolver.
*  Performance improvement in Transform by avoiding excess encodings and
   decodings when it materializes transformed examples or generates statistics
   (both enabled by default).
*  Added an accessor (`.data_view_decode_fn`) to the decoder function wrapped in
   the DataView in Trainer `FnArgs.data_accessor`.

## Breaking Changes

*   Starting in this version, following artifacts will be stored in new format,
    but artifacts produced by older versions can be read in a backwards
    compatible way:
    *   Change split sub-folder format to 'Split-<split_name>', this applies to
        all artifacts that contain splits. Old format '<split_name>' can still
        be loaded by TFX.
    *   Change Model artifact's sub-folder name to 'Format-TFMA' for eval model
        and 'Format-Serving' for serving model. Old Model artifact format
        ('eval_model_dir'/'serving_model_dir') can still be loaded by TFX.
    *   Change ExampleStatistics artifact payload to binary proto
        FeatureStats.pb file. Old payload format (tfrecord stats_tfrecord file)
        can still be loaded by TFX.
    *   Change ExampleAnomalies artifact payload to binary proto SchemaDiff.pb
        file. Old payload format (text proto anomalies.pbtxt file) is deprecated
        as TFX doesn't have downstream components that take ExampleAnomalies
        artifact.


### For Pipeline Authors

*  CLI requires Apache Airflow 1.10.14 or later. If you are using an older
   version of airflow, you can still copy runner definition to the DAG
   directory manually and run using airflow UIs.

### For Component Authors

*   N/A

## Deprecations

*   Deprecated input/output compatibility aliases for Transform and
    StatisticsGen.

## Bug Fixes and Other Changes

*   The `tfx_version` custom property of output artifacts is now set by the
    default publisher to the TFX SDK version.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `kfp-pipeline-spec>=0.1.7,<0.2`.
*   Depends on `ml-metadata>=0.29.0,<0.30.0`.
*   Depends on `packaging>=20,<21`.
*   Depends on `struct2tensor>=0.29.0,<0.30.0`.
*   Depends on `tensorflow-data-validation>=0.29.0,<0.30.0`.
*   Depends on `tensorflow-model-analysis>=0.29.0,<0.30.0`.
*   Depends on `tensorflow-transform>=0.29.0,<0.30.0`.
*   Depends on `tfx-bsl>=0.29.0,<0.30.0`.

## Documentation Updates

*   Simplified Apache Spark and Flink example deployment scripts by using Beam's
    SparkRunner and FlinkRunner classes.
*   Upgraded example Apache Flink deployment to Flink 1.12.1.
*   Upgraded example Apache Spark deployment to Spark 2.4.7.
*   Added the "TFX Python function component" notebook tutorial.
