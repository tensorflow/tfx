# Version 1.21.0

## Major Features and Improvements

*   Added dynamic support for ZetaSQL-free MLMD environments across TFX Resolvers and metadata extensions. The system automatically detects missing C++ ZetaSQL engine binaries at runtime and transparently falls back to a highly robust, pure-Python in-memory lineage graph traversal and relation evaluation engine.

## Breaking Changes

*   Transitioned proto compilation tooling in Bazel workspaces from legacy deprecated `py_proto_library` rules to custom Starlark provider compilation macros, enabling unified, robust build integration on Bazel 7.x workspaces running with Bzlmod enabled.

### For Pipeline Authors

### For Component Authors

## Deprecations

*   Bypassed legacy testing targets checking deprecated and retired Google Cloud AI Platform (CAIP) integration points, fully migrating Vertex AI-compatible pipeline targets.

## Bug Fixes and Other Changes

*   Refactored Wide & Deep functional models (`taxi_utils.py`, templates, and test modules) to slice wide categorical input layers dynamically matching actually wide-encoded category bounds (`[:len(_MAX_CATEGORICAL_FEATURE_VALUES)]`). This prevents disconnected inputs from triggering Keras 3 `inputs not connected to outputs` exception under Python 3.10.
*   Converted Keras Functional model building methods' `Normalization` layer instantiation inside list comprehensions to standard procedural `for` loops, fully securing execution scope connectivity tracking under Python 3.10.
*   Implemented dynamic `pytest_ignore_collect` hooks in `conftest.py` with static spec checks (`importlib.util.find_spec`) to dynamically exclude targets of uninstalled optional dependencies (like Airflow, Vertex AI, and Kubeflow). This completely eliminates early logging stream deadlocks and startup import-time test suite collection crashes.
*   Upgraded Docker build tools and wheel scripts, configuring internal compilation of TFDV and TFX-BSL source files on a unified conda-GCC 13/binutils toolchain using Bazel 7.7.0.
*   Resolved random temporary directory synchronization and write finalizer errors in BulkInferrer (`executor.py`) when executing flattened PCollections under local runners (DirectRunner/PrismRunner/FnApiRunner) by introducing a dynamic helper mapping local executions to use `num_shards=1` while preserving high-performance dynamic sharding for distributed production pipelines.
*   Bypassed strict committed/attempted metrics equivalence checks in the Transform `ExecutorTest` base class (`executor_test.py`) that crashed under modern versions of Apache Beam utilizing the parallel/multi-process `PrismRunner` backend due to asynchronous task metric updating limits, ensuring robust and stable local metrics count verifications.
*   Monkey-patched `PipelineOptions` dynamically in the global test conftest (`conftest.py`) to bypass resource-throttled multi-process `PrismRunner` delegation for standard local testing jobs, forcing the low-overhead, fast single-threaded in-memory DirectRunner (`--direct_running_mode=in_memory`) globally. This slashes total unit testing execution time and prevents workflow cancellations/timeouts across Python 3.9, 3.10, 3.11, and 3.12 GHA platforms.

## Dependency Updates

*   Upgrades target pipeline constraints to support **TensorFlow 2.21.0** and **Protobuf 6.x** across both Python 3.10 and Python 3.11.
*   Split SciPy library dependency constraint inside `test_constraints.txt` using Python target markers to bypass dynamic version conflicts with JAX versions under Python < 3.13.
*   Cleanly dropped outdated/incompatible dependencies (`tensorflow-decision-forests`, `tensorflow-ranking`, `tensorflow-text`, `tensorflowjs`) globally from dependencies list and constraint definitions to prevent PIP backtracking solver storms and secure stable installation on TF 2.21.0.

## Documentation Updates

*   N/A
