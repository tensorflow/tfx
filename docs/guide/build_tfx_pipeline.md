# Building TFX pipelines

!!! Note
    For a conceptual view of TFX Pipelines, see
    [Understanding TFX Pipelines](understanding_tfx_pipelines.md).

!!!Note
    Want to build your first pipeline before you dive into the details? Get
    started
    [building a pipeline using a template](build_local_pipeline.md#build-a-pipeline-using-a-template).

## Using the `Pipeline` class

TFX pipelines are defined using the
[`Pipeline` class](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external }.
The following example demonstrates how to use the `Pipeline` class.

```python
pipeline.Pipeline(
    pipeline_name=pipeline-name,
    pipeline_root=pipeline-root,
    components=components,
    enable_cache=enable-cache,
    metadata_connection_config=metadata-connection-config,
)
```

Replace the following:

*   `pipeline-name`: The name of this pipeline. The pipeline name must
    be unique.

    TFX uses the pipeline name when querying ML Metadata for component input
    artifacts. Reusing a pipeline name may result in unexpected behaviors.

*   `pipeline-root`: The root path of this pipeline's outputs. The root
    path must be the full path to a directory that your orchestrator has read
    and write access to. At runtime, TFX uses the pipeline root to generate
    output paths for component artifacts. This directory can be local, or on a
    supported distributed file system, such as Google Cloud Storage or HDFS.

*   `components`: A list of component instances that make up this
    pipeline's workflow.

*   `enable-cache`: (Optional.) A boolean value that indicates if this
    pipeline uses caching to speed up pipeline execution.

*   `metadata-connection-config`: (Optional.) A connection
    configuration for ML Metadata.

## Defining the component execution graph

Component instances produce artifacts as outputs and typically depend on
artifacts produced by upstream component instances as inputs. The execution
sequence for component instances is determined by creating a directed acyclic
graph (DAG) of the artifact dependencies.

For instance, the `ExampleGen` standard component can ingest data from a CSV
file and output serialized example records. The `StatisticsGen` standard
component accepts these example records as input and produces dataset
statistics. In this example, the instance of `StatisticsGen` must follow
`ExampleGen` because `SchemaGen` depends on the output of `ExampleGen`.

### Task-based dependencies

!!! Note
    Using task-based dependencies is typically not recommended. Defining the
    execution graph with artifact dependencies lets you take advantage of the
    automatic artifact lineage tracking and caching features of TFX.

You can also define task-based dependencies using your component's
[`add_upstream_node` and `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external }
methods. `add_upstream_node` lets you specify that the current component must be
executed after the specified component. `add_downstream_node` lets you specify
that the current component must be executed before the specified component.

## Pipeline templates

The easiest way to get a pipeline set up quickly, and to see how all the pieces
fit together, is to use a template. Using templates is covered in [Building a
TFX Pipeline Locally](../build_local_pipeline).

## Caching

TFX pipeline caching lets your pipeline skip over components that have been
executed with the same set of inputs in a previous pipeline run. If caching is
enabled, the pipeline attempts to match the signature of each component, the
component and set of inputs, to one of this pipeline's previous component
executions. If there is a match, the pipeline uses the component outputs from
the previous run. If there is not a match, the component is executed.

Do not use caching if your pipeline uses non-deterministic components. For
example, if you create a component to create a random number for your pipeline,
enabling the cache causes this component to execute once. In this example,
subsequent runs use the first run's random number instead of generating a random
number.
