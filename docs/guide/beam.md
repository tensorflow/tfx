# Apache Beam and TFX

[Apache Beam](https://beam.apache.org/) provides a framework for running batch
and streaming data processing jobs that run on a variety of execution engines.
Several of the TFX libraries use Beam for running tasks, which enables a high
degree of scalability across compute clusters.  Beam includes support for a
variety of execution engines or "runners", including a direct runner which runs
on a single compute node and is very useful for development, testing, or small
deployments.  Beam provides an abstraction layer which enables TFX to run on any
supported runner without code modifications.  TFX uses the Beam Python API, so
it is limited to the runners that are supported by the Python API.

## Deployment and Scalability

As workload requirements increase Beam can scale to very large deployments
across large compute clusters. This is limited only by the scalability of the
underlying runner.  Runners in large deployments will typically be deployed to a
container orchestration system such as Kubernetes or Apache Mesos for automating
application deployment, scaling, and management.

See the [Apache Beam](https://beam.apache.org/) documentation for more
information on Apache Beam.

For Google Cloud users, [Dataflow](https://cloud.google.com/dataflow) is the
recommended runner, which provides a serverless and cost-effective platform
through autoscaling of resources, dynamic work rebalancing, deep integration
with other Google Cloud services, built-in security, and monitoring.

## Custom Python Code and Dependencies

One notable complexity of using Beam in a TFX pipeline is handling custom code
and/or the dependencies needed from additional Python modules. Here are some
examples of when this might be an issue:

*   preprocessing_fn needs to refer to the user's own Python module
*   a custom extractor for the Evaluator component
*   custom modules which are sub-classed from a TFX component

TFX relies on Beam's support for
[Managing Python Pipeline Dependencies](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)
to handle Python dependencies. Currently there are two ways to manage this:

1.  Providing Python Code and Dependencies as Source Package
1.  [Dataflow only] Using a Container Image as Worker

These are discussed next.

### Providing Python Code and Dependencies as a Source Package

This is recommended for users who:

1.  Are familiar with Python packaging and
1.  Only use Python source code (i.e., no C modules or shared libraries).

Please follow one of the paths in
[Managing Python Pipeline Dependencies](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)
to provide this using one of the following beam_pipeline_args:

*   --setup_file
*   --extra_package
*   --requirements_file

Notice: In any of above cases, please make sure that the same version of `tfx`
is listed as a dependency.

### [Dataflow only] Using a Container Image for a Worker

TFX 0.26.0 and above has experimental support for using
[custom container image](https://beam.apache.org/documentation/runtime/environments/#customizing-container-images)
for Dataflow workers.

In order to use this, you have to:

*   Build a Docker image which has both `tfx` and the users' custom code and
    dependencies pre-installed.
    *   For users who (1) use `tfx>=0.26` and (2) uses python 3.7 to develop their pipelines,
        the easiest way to do this is extending the corresponding version of the official
        `tensorflow/tfx` image:

```Dockerfile
# You can use a build-arg to dynamically pass in the
# version of TFX being used to your Dockerfile.

ARG TFX_VERSION
FROM tensorflow/tfx:${TFX_VERSION}
# COPY your code and dependencies in
```

*   Push the image built to a container image registry which is accessible by
    the project used by Dataflow.
    *   Google Cloud users can consider using
        [Cloud Build](https://cloud.google.com/cloud-build/docs/quickstart-build)
        which nicely automates above steps.
*   Provide following `beam_pipeline_args`:

```python
beam_pipeline_args.extend([
    '--runner=DataflowRunner',
    '--project={project-id}',
    '--worker_harness_container_image={image-ref}',
    '--experiments=use_runner_v2',
])
```

**TODO(b/171733562): Remove use_runner_v2 once it is default for Dataflow.**

**TODO(b/179738639): Create documentation for how to test custom container
locally after https://issues.apache.org/jira/browse/BEAM-5440.**

## Beam Pipeline Arguments

Several TFX components rely on Beam for distributed data processing. They are
configured with `beam_pipeline_args`, which is specified during during pipeline
creation:

```python
my_pipeline = Pipeline(
    ...,
    beam_pipeline_args=[...])
```

TFX 0.30 and above adds an interface, `with_beam_pipeline_args`, for extending
the pipeline level beam args per component:

```python
example_gen = CsvExampleGen(input_base=data_root).with_beam_pipeline_args([...])
```
