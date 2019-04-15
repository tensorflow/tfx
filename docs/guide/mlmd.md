# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) is a library
for recording and retrieving metadata
associated with ML developer and data scientist workflows. MLMD is an integral
part of [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), but is
designed so that it can be used independently. As part of a broader platform
like TFX, most users only interact with MLMD when examining the results of
pipeline components, for example in notebooks or in TensorBoard.

The graph below shows the components that are part of MLMD. The storage backend
is pluggable and can be extended. MLMD provides reference implementations for
SQLite (which supports in-memory and disk) and MySQL out of the box. The
MetadataStore provides APIs to record and retrieve metadata to and from the
storage backend. MLMD can register metadata about the artifacts generated
through the components/steps of the pipelines, metadata about the executions of
these components/steps, and the associated lineage information. The concepts are
explained in more detail below.

![ML Metadata Overview](images/mlmd_overview.png)

## Functionality Enabled by MLMD

Tracking the inputs and outputs of all components/steps in an ML workflow and
their lineage allows ML platforms to enable several important features. The
following list provides a non-exhaustive overview of some of the major benefits.

*   **List all Artifacts of a specific type**, e.g. all Models that have been
    trained.
*   **Load two Artifacts of the same type for comparison**, e.g. to compare
    results from two experiments.
*   **Show a DAG of all executions and their input and output artifacts**, e.g.
    to visualize the workflow for debugging and discovery.
*   **Recurse back through all events to see how an artifact was created**, e.g.
    to see what data went into a model, or to enforce data retention plans.
*   **Identify all artifacts that were created using a given artifact**, e.g. to
    see all Models trained from a specific dataset, to mark models based upon
    bad data.
*   **Determine if an execution has been run on the same inputs before**, e.g.
    to determine whether a component/step has already completed the same work
    and the previous output can just be reused.
*   Etc.

## Metadata Storage Backends and Store Connection Configuration

The MetadataStore object receives a connection configuration that corresponds to
the storage backend used.

*   **Fake Database** provides an in-memory DB (using SQLite) for fast
    experimentation and local runs. Database is deleted when store object is
    destroyed.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Empty fake database proto
store = metadata_store.MetadataStore(connection_config)
```

*   **SQLite** reads and writes files from disk.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '...'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = metadata_store.MetadataStore(connection_config)
```

*   **MySQL** connects to a MySQL server.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.mysql.host = '...'
connection_config.mysql.port = '...'
connection_config.mysql.database = '...'
connection_config.mysql.user = '...'
connection_config.mysql.password = '...'
store = metadata_store.MetadataStore(connection_config)
```
