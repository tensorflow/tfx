# Version 1.3.0

## Major Features and Improvements

*   TFX CLI now supports runtime parameter on Kubeflow, Vertex, and Airflow.
    Use it with '--runtime_parameter=<parameter_name>=<parameter_value>' flag.
    In the case of multiple runtime parameters, format is as follows:
    '--runtime_parameter=<parameter_name>=<parameter_value> --runtime_parameter
    =<parameter_name>=<parameter_value>'
*   Added Manual node in the experimental orchestrator.
*   Placeholders support index access and JSON serialization for list type execution properties.
*   Added `ImportSchemaGen` which is a dedicated component to import a
    pre-defined schema file. ImportSchemaGen will replace `Importer` with
    simpler syntax and less constraints. You have to pass the file path to the
    schema file instead of the parent directory unlike `Importer`.
*   Added support for outputting and encoding `tf.RaggedTensor`s in TFX
    Transform component.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   The import name of KerasTuner has been changed from `kerastuner`
    to `keras_tuner`. The import name of `kerastuner` is still supported.
    A warning will occur when import from `kerastuner`, but does not affect
    the usage.

## Bug Fixes and Other Changes

*   The default job name for Google Cloud AI Training jobs was changed from
    'tfx_YYYYmmddHHMMSS' to 'tfx_YYYYmmddHHMMSS_xxxxxxxx', where 'xxxxxxxx' is
    a random 8 digit hexadecimal string.
*   Fix component to raise error if its input required channel (specified from
    ComponentSpec) has no artifacts in it.
*   Fixed an issue where ClientOptions with regional endpoint was
    incorrectly left out in Vertex AI pusher.
*   CLI now hides passed flags from user python files in "--pipeline-path". This
    will prevent errors when user python file tries reading and parsing flags.
*   Fixed missing type information marker file 'py.typed'.
*   Depends on `apache-beam[gcp]>=2.32,<3`.
*   Depends on `google-cloud-bigquery>=1.28.0,<3`.
*   Depends on `jinja2>=2.7.3,<4`, i.e. now supports Jinja 3.x.
*   Depends on `keras-tuner>=1.0.4,<2`.
*   Depends on `kfp>=1.6.1,!=1.7.2,<1.8.2` in \[kfp\] extra.
*   Depends on `kfp-pipeline-spec>=>=0.1.10,<0.2`.
*   Depends on `ml-metadata>=1.3.0,<1.4.0`.
*   Depends on `struct2tensor>=0.34.0,<0.35.0`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on `tensorflow-data-validation>=1.3.0,<1.4.0`.
*   Depends on `tensorflow-model-analysis>=0.34.1,<0.35.0`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on `tensorflow-transform>=1.3.0,<1.4.0`.
*   Depends on `tfx-bsl>=1.3.0,<1.4.0`.

## Documentation Updates

*   N/A
