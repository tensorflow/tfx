# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for input resolution.

When running a node (e.g. Pusher, Resolver, ...) in the TFX pipeline, its input
is first fetched by the input channel definition. This initial input dict
(type: Dict[str, List[Artifact]]) goes through the *input resolution* process
which transforms an initial input dict to a list of input dicts. These
*resolved inputs* are then executed by running Executor.Do() for each input
dict.

Typically `Resolver` node specifies the input resolution logic (other types
of node simply returns `[input_dict]` during input resolution). Input resolution
logic of a node can be specified in two ways:

## 1. Using a single `ResolverStrategy`.

This is the classical way of specifying input resolution logic only for
`Resolver` node. Subclass of ResolverStrategy implements `resolve_artifacts()`
method which takes an input dict and produce an optional input dict. If the
result of `resolve_artifacts()` is not None, input resolution result is
a list of that single input dict (`[result]`). Else if the result is None,
input resolution result is an empty list (`[]`).

`Resolver` defines its input resolution logic by specifying `ResolverStrategy`
class on its node creation.

```python
my_resolver = tfx.dsl.Resolver(
    strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
    examples=example_gen.outputs['examples']
).with_id('my_resolver')
```

## 2. Using a `@resolver` decorated function.

Decorated function (let's call it a *resolver function*) can express complex
input resolution logic beyond a single `ResolverStrategy` by combining multiple
`ResolverOp`s. Each `ResolverOp` represents a single function, and its output
can be fed into inputs of other `ResolverOp`s. Final return value of the
function should be a dict or a list of dict. Signature of `ResolverStrategy` is
always dict -> dict, but signature of `ResolverOp` is more flexible.

For convenience, we allow `ResolverStrategy` to be used in place where
`ResolverOp` can be used. Consider `ResolverStrategy` as a special kind of
`ResolverOp` with dict -> dict signature (though there isn't any inheritance).
The other way is not compatible; `ResolverOp` cannot be used in strategy_class=
argument for `Resolver` node.

```python
@resolver
class MyResolver(input_dict):
  # ResolverOp is a building block of input resolution logic.
  result = MyCustomResolverOp(input_dict, flag=False)
  # You can even use ResolverStrategy as a building block just like ResolverOp.
  result = LatestArtifactStrategy(result)
  return result
```

Note that invoking `ResolverOp(input_dict)` doesn't create a `ResolverOp`
instance, but a dummy object (`OpNode`) for function tracing.

`@resolver` decorator converts this *resolver function* to a `Resolver` node
factory, so that calling this function would create a `Resolver` node.

```python
my_resolver = MyResolver(
    examples=example_gen.outputs['examples']
).with_id('my_resolver')
```

Internally `@resolver` decorator wraps decorated function to a
`ResolverFunction` object, and pass it to the `Resolver` node arguments.
`ResolverFunction` object is not a user facing API.
"""
