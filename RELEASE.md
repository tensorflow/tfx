# Version 1.15.0

## Major Features and Improvements

*  Dropped python 3.8 support.
*  Extend GetPipelineRunExecutions, GetPipelineRunArtifacts APIs to support
   filtering by execution create_time, type.
*  ExampleValidator and DistributionValidator now support anomalies alert
   generation. Users can use their own toolkits to extract and process the
   alerts from the execution parameter.
*  Allow DistributionValidator baseStatistics input channel artifacts to be
   empty for cold start of data validation.
*  `ph.make_proto()` allows constructing proto-valued placeholders, e.g. for
   larger config protos fed to a component.
*  `ph.join_path()` is like `os.path.join()` but for placeholders.
*  Support passing in `experimental_debug_stripper` into the Transform
   pipeline runner.

## Breaking Changes

*   `Placeholder` and all subclasses have been moved to other modules, their
    structure has been changed and they're now immutable. Most users won't care
    (the main public-facing API is unchanged and behaves the same way). If you
    do special operations like `isinstance()` or some kind of custom
    serialization on placeholders, you will have to update your code.
*   `placeholder.Placeholder.traverse()` now returns more items than before,
    namely also placeholder operators like `_ConcatOperator` (which is the
    implementation of Python's `+` operator).
*   The `placeholder.RuntimeInfoKey` enumeration was removed. Just hard-code the
    appropriate string values in your code, and reference the new `Literal` type
    `placeholder.RuntimeInfoKeys` if you want to ensure correctness.
*   Arguments to `@component` must now be passed as kwargs and its return type
    is changed from being a `Type` to just being a callable that returns a new
    instance (like the type's initializer). This will allow us to instead return
    a factory function (which is not a `Type`) in future. For a given
    `@component def C()`, this means:
    *   You should not use `C` as a type anymore. For instance, replace
        `isinstance(foo, C)` with something else. Depending on your use case, if
        you just want to know whether it's a component, then use
        `isinstance(foo, tfx.types.BaseComponent)` or
        `isinstance(foo, tfx.types.BaseFunctionalComponent)`.
        If you want to know _which_ component it is, check its `.id` instead.
        Existing such checks will break type checking today and may additionally
        break at runtime in future, if we migrate to a factory function.
    *   You can continue to use `C.test_call()` like before, and it will
        continue to be supported in future.
    *   Any type declarations using `foo: C` break and must be replaced with
        `foo: tfx.types.BaseComponent` or
        `foo: tfx.types.BaseFunctionalComponent`.
    *   Any references to static class members like `C.EXECUTOR_SPEC` breaks
        type checking today and should be migrated away from. In particular, for
        `.EXECUTOR_SPEC.executor_class().Do()` in unit tests, use `.test_call()`
        instead.
    *   If your code previously asserted a wrong type declaration on `C`, this
        can now lead to (justified) type checking errors that were previously
        hidden due to `C` being of type `Any`.
*   `ph.to_list()` was renamed to `ph.make_list()` for consistency.


### For Pipeline Authors

### For Component Authors

## Deprecations

*   Deprecated python 3.8

## Bug Fixes and Other Changes

* Fixed a synchronization bug in google_cloud_ai_platform tuner.
* Print best tuning trials only from the chief worker of google_cloud_ai_platform tuner.
* Add a kpf dependency in the docker-image extra packages.
* Fix BigQueryExampleGen failure without custom_config.

## Dependency Updates
| Package Name | Version Constraints | Previously (in `v1.14.0`) | Comments |
| -- | -- | -- | -- |
| `keras-tuner` | `>=1.0.4,<2,!=1.4.0,!=1.4.1` | `>=1.0.4,<2` | |
| `packaging` | `>=20,<21` | `>=22` | |
| `attrs` | `19.3.0,<22` | `19.3.0,<24` | |
| `google-cloud-bigquery` | `>=2.26.0,<3` | `>=3,<4` | |
| `tensorflow` | `>=2.15,<2.16` | `>=2.13,<2.14` | |
| `tensorflow-decision-forests` | `>=1.0.1,<1.9` | `>=1.0.1,<2` | |
| `tensorflow-hub` | `>=0.9.0,<0.14` | `>=0.15.0,<0.16` | |
| `tensorflow-serving` | `>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,<3` | `>=2.15,<2.16` | |

## Documentation Updates
