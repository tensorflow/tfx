# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Proto library helper utils."""

# Custom provider for descriptor proto files.
ProtoDescriptorInfo = provider(
    fields = {
        "direct_sources": "Direct proto source files",
        "transitive_sources": "Transitive proto source files",
    },
)

# Helper rule to make proto files available without compilation.
def _proto_descriptor_impl(ctx):
    proto_sources = depset(direct = ctx.files.srcs)
    return [
        DefaultInfo(files = proto_sources),
        ProtoDescriptorInfo(
            direct_sources = ctx.files.srcs,
            transitive_sources = proto_sources,
        ),
    ]

proto_descriptor = rule(
    implementation = _proto_descriptor_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".proto"]),
    },
)

def _py_proto_library_impl(ctx):
    proto_deps = ctx.attr.deps
    workspace_root = ctx.attr.workspace_root
    use_grpc_plugin = ctx.attr.use_grpc_plugin

    all_sources = []
    py_infos = []
    all_proto_infos = []

    for dep in proto_deps:
        if ProtoInfo in dep:
            all_sources.extend(dep[ProtoInfo].direct_sources)
            all_proto_infos.append(dep[ProtoInfo])
        elif ProtoDescriptorInfo in dep:
            all_sources.extend(dep[ProtoDescriptorInfo].direct_sources)
        elif PyInfo in dep:
            py_infos.append(dep[PyInfo])

    workspace_sources = []
    for src in all_sources:
        if not src.short_path.startswith("external/") and not src.short_path.startswith("../"):
            workspace_sources.append(src)

    # Find workspace root and current source directory
    # workspace_root is passed as parameter from macro or --define
    if not workspace_root:
        workspace_root = ctx.var.get("TFX_ROOT", "")

    # If not provided, derive from workspace sources
    if not workspace_root:
        for src in workspace_sources:
            if src.path.startswith("/"):
                parts = src.path.split("/bazel-out/")
                if len(parts) == 2:
                    workspace_root = parts[0]
                    break

    def _make_abs(path):
        if workspace_root and not path.startswith("/"):
            return workspace_root + "/" + path
        return path

    # Use relative external path for sandbox execution
    bazel_tfx_root = "external"

    # Declare outputs with consistent naming
    py_outputs = []
    pb2_outputs = {}  # Map proto file basename to _pb2.py output
    pb2_grpc_outputs = {}  # Map proto file basename to _pb2_grpc.py output

    for proto_src in workspace_sources:
        basename = proto_src.basename[:-6]
        pb2_file = ctx.actions.declare_file(basename + "_pb2.py")
        py_outputs.append(pb2_file)
        pb2_outputs[basename] = pb2_file

        if use_grpc_plugin:
            pb2_grpc_file = ctx.actions.declare_file(basename + "_pb2_grpc.py")
            py_outputs.append(pb2_grpc_file)
            pb2_grpc_outputs[basename] = pb2_grpc_file

    if py_outputs:
        proto_path_args = ["--proto_path=."]
        proto_paths = {".": True}

        # Collect all proto files from transitive dependencies
        all_transitive_sources = []
        for dep in proto_deps:
            if ProtoInfo in dep:
                all_transitive_sources.extend(dep[ProtoInfo].transitive_sources.to_list())

        # Add proto paths from workspace sources
        for ws in workspace_sources:
            ws_dir = "/".join(ws.short_path.split("/")[:-1])
            if ws_dir and ws_dir not in proto_paths:
                proto_paths[ws_dir] = True
                proto_path_args.append("--proto_path=" + ws_dir)

        # Add proto paths from external dependencies under external/
        protobuf_proto_path = bazel_tfx_root + "/com_google_protobuf/src"
        if protobuf_proto_path not in proto_paths:
            proto_paths[protobuf_proto_path] = True
            proto_path_args.append("--proto_path=" + protobuf_proto_path)

        mlmd_proto_path = bazel_tfx_root + "/com_github_google_ml_metadata"
        if mlmd_proto_path not in proto_paths:
            proto_paths[mlmd_proto_path] = True
            proto_path_args.append("--proto_path=" + mlmd_proto_path)

        # Add proto paths from all transitive sources (including external dependencies)
        for src in all_transitive_sources:
            src_path = src.path

            if "bazel-out/" in src_path:
                # Skip bazel-out paths; we use bazel-tfx instead.
                continue

            # Add proto path from the directory containing each file
            src_dir = "/".join(src_path.split("/")[:-1])
            if src_dir and src_dir not in proto_paths:
                # For external dependencies, check if there's a known proto root subdirectory
                if "external/" in src_dir:
                    parts = src_dir.split("external/")
                    rest = parts[1]  # e.g., "com_google_protobuf/python/google/protobuf"
                    rest_parts = rest.split("/")
                    package_name = rest_parts[0]  # e.g., "com_google_protobuf"

                    # Build the proto root, including known subdirectories
                    proto_root = bazel_tfx_root + "/" + package_name

                    if len(rest_parts) > 1:
                        next_part = rest_parts[1]
                        # Include subdirectories like python/, src/, proto/ that contain actual proto files
                        if next_part in ["python", "src", "proto", "protobuf", "include"]:
                            proto_root = proto_root + "/" + next_part

                    if proto_root not in proto_paths:
                        proto_paths[proto_root] = True
                        proto_path_args.append("--proto_path=" + proto_root)
                elif src_dir not in proto_paths:
                    proto_paths[src_dir] = True
                    proto_path_args.append("--proto_path=" + src_dir)

        proto_file_args = [src.short_path for src in workspace_sources]
        protoc_args = ["--python_out=" + ctx.bin_dir.path]
        tools = []

        if use_grpc_plugin:
            protoc_args.append("--grpc_python_out=" + ctx.bin_dir.path)
            protoc_args.append(
                "--plugin=protoc-gen-grpc_python=" + ctx.executable._grpc_python_plugin.path
            )
            tools.append(ctx.executable._grpc_python_plugin)

        ctx.actions.run(
            inputs = depset(
                direct = workspace_sources,
                transitive = [
                    dep[ProtoInfo].transitive_sources
                    for dep in proto_deps
                    if ProtoInfo in dep
                ],
            ),
            outputs = py_outputs,
            executable = ctx.executable._protoc,
            arguments = protoc_args + proto_path_args + proto_file_args,
            mnemonic = "ProtocPython",
            execution_requirements = {"no-sandbox": "1"},
            tools = tools,
        )

        if workspace_root:
            copy_stamp = ctx.actions.declare_file(ctx.label.name + "_python_out_copy.stamp")
            copy_lines = ["set -e"]

            for output_file in py_outputs:
                rel_path = output_file.path
                bin_dir_prefix = ctx.bin_dir.path + "/"
                if rel_path.startswith(bin_dir_prefix):
                    rel_path = rel_path[len(bin_dir_prefix):]
                else:
                    rel_path = output_file.basename

                dest_path = workspace_root + "/" + rel_path
                dest_dir = "/".join(dest_path.split("/")[:-1])
                copy_lines.append("mkdir -p \"" + dest_dir + "\"")
                copy_lines.append("cp -f \"" + output_file.path + "\" \"" + dest_path + "\"")

            copy_lines.append("touch \"" + copy_stamp.path + "\"")

            ctx.actions.run_shell(
                inputs = py_outputs,
                outputs = [copy_stamp],
                command = "\n".join(copy_lines),
                mnemonic = "CopyPythonOut",
            )

    all_transitive_sources = [depset(py_outputs)]
    all_imports = [depset([ctx.bin_dir.path])] if py_outputs else []

    for py_info in py_infos:
        all_transitive_sources.append(py_info.transitive_sources)
        if hasattr(py_info, "imports"):
            all_imports.append(py_info.imports)

    # Store individual file outputs in output groups for easy access
    output_groups = {}
    for basename, output_file in pb2_outputs.items():
        output_groups[basename + "_pb2"] = depset([output_file])
    for basename, output_file in pb2_grpc_outputs.items():
        output_groups[basename + "_pb2_grpc"] = depset([output_file])

    return [
        DefaultInfo(files = depset(py_outputs)),
        PyInfo(
            transitive_sources = depset(transitive = all_transitive_sources),
            imports = depset(transitive = all_imports),
            has_py2_only_sources = False,
            has_py3_only_sources = True,
        ),
        OutputGroupInfo(**output_groups),
    ]

_py_proto_library_rule = rule(
    implementation = _py_proto_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [[ProtoInfo], [PyInfo]],
        ),
        "use_grpc_plugin": attr.bool(
            default = False,
            doc = "Enable gRPC Python stub generation",
        ),
        "workspace_root": attr.string(
            default = "",
            doc = "Workspace root directory path",
        ),
        "_protoc": attr.label(
            default = "@com_google_protobuf//:protoc",
            executable = True,
            cfg = "exec",
        ),
        "_grpc_python_plugin": attr.label(
            default = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [PyInfo],
)

def tfx_py_proto_library(name, srcs = [], deps = [], visibility = None, testonly = 0, use_grpc_plugin = False, workspace_root = None):
    """Opensource py proto generation using a custom rule."""
    if not srcs:
        fail("srcs must not be empty for tfx_py_proto_library")

    if workspace_root == None:
        workspace_root = ""

    native.proto_library(
        name = name + "_proto",
        srcs = srcs,
        testonly = testonly,
        visibility = visibility,
    )
    _py_proto_library_rule(
        name = name,
        deps = [":" + name + "_proto"] + deps,
        use_grpc_plugin = use_grpc_plugin,
        workspace_root = workspace_root,
        visibility = ["//visibility:public"],  # Must be public for aliases to be accessible
        testonly = testonly,
    )

    # Create individual file aliases for backward compatibility
    # These allow targets like //tfx/proto:bulk_inferrer_pb2.py to work
    for src in srcs:
        if src.endswith(".proto"):
            basename = src[:-6]  # Remove .proto extension
            native.alias(
                name = basename + "_pb2.py",
                actual = ":" + name,
                visibility = ["//visibility:public"],  # Aliases must be public for cross-package access
                testonly = testonly,
            )
            if use_grpc_plugin:
                native.alias(
                    name = basename + "_pb2_grpc.py",
                    actual = ":" + name,
                    visibility = ["//visibility:public"],  # Aliases must be public for cross-package access
                    testonly = testonly,
                )
