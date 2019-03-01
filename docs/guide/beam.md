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
