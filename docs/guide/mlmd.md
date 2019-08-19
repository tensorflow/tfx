# ML Metadata

[ML Metadata (MLMD)](https://github.com/google/ml-metadata) is a library for
recording and retrieving metadata associated with ML developer and data
scientist workflows. MLMD is an integral part of
[TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx), but is designed so
that it can be used independently. As part of the broader TFX platform, most
users only interact with MLMD when examining the results of pipeline components,
for example in notebooks or in TensorBoard.

The graph below shows the components that are part of MLMD. The storage backend
is pluggable and can be extended. MLMD provides reference implementations for
SQLite (which supports in-memory and disk) and MySQL out of the box. The
MetadataStore provides APIs to record and retrieve metadata to and from the
storage backend. MLMD can register:

-   metadata about the artifacts generated through the components/steps of the
    pipelines
-   metadata about the executions of these components/steps
-   metadata about the pipeline and associated lineage information

The concepts are explained in more detail below.

![ML Metadata Overview](images/mlmd_overview.png)

## Functionality Enabled by MLMD

Tracking the inputs and outputs of all components/steps in an ML workflow and
their lineage allows ML platforms to enable several important features. The
following list provides a non-exhaustive overview of some of the major benefits.

*   **List all Artifacts of a specific type.** Example: all Models that have
    been trained.
*   **Load two Artifacts of the same type for comparison.** Example: compare
    results from two experiments.
*   **Show a DAG of all related executions and their input and output artifacts
    of a context.** Example: visualize the workflow of an experiment for
    debugging and discovery.
*   **Recurse back through all events to see how an artifact was created.**
    Examples: see what data went into a model; enforce data retention plans.
*   **Identify all artifacts that were created using a given artifact.**
    Examples: see all Models trained from a specific dataset; mark models based
    upon bad data.
*   **Determine if an execution has been run on the same inputs before.**
    Example: determine whether a component/step has already completed the same
    work and the previous output can just be reused.
*   **Record and query context of workflow runs.** Examples: track the owner and
    changelist used for a workflow run; group the lineage by experiments; manage
    artifacts by projects.

## Metadata Storage Backends and Store Connection Configuration

The MetadataStore object receives a connection configuration that corresponds to
the storage backend used.

*   **Fake Database** provides an in-memory DB (using SQLite) for fast
    experimentation and local runs. Database is deleted when store object is
    destroyed.

```python
connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent() # Sets an empty fake database proto.
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

### Upgrade MLMD library

When using a new MLMD release or your own build with an existing MLMD database,
there may be database schema changes. Unless a breaking change is explicitly
mentioned in the release note, all MLMD database schema changes are transparent
for the MLMD API users.

When the MLMD library connects to the database, it compares the expected schema
version of the MLMD library (`library_version`) with the schema version
(`db_version`) recorded in the given database.

*   If `library_version` is compatible with `db_version`, nothing happens.
*   If `library_version` is newer than `db_version`, it runs a single migration
    transaction to evolve the database by executing a series of migration
    scripts. The migration script is provided together with any schema change
    commit and enforced by change reviews and verified by continuous tests.

    NOTE: If the migration transaction has errors, the transaction will rollback
    and keep the original database unchanged. If this case happens, it is
    possible that other concurrent transaction ran in the same database, or the
    migration script's test fails to capture all cases. If it is the latter
    case, please report issues for a fix or downgrade library to work with the
    database.

*   If `library_version` is older than `db_version`, MLMD library returns errors
    to prevent any data loss. In this case, the user should upgrade the library
    version before using that database.
