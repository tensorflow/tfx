package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

proto_library(
    name = "bulk_inferrer_proto",
    srcs = ["bulk_inferrer.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "bulk_inferrer_proto_py_pb2",
    api_version = 2,
    deps = [":bulk_inferrer_proto"],
)

proto_library(
    name = "evaluator_proto",
    srcs = ["evaluator.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "evaluator_proto_py_pb2",
    api_version = 2,
    deps = [":evaluator_proto"],
)

proto_library(
    name = "example_gen_proto",
    srcs = ["example_gen.proto"],
    cc_api_version = 2,
    deps = ["//google/protobuf:any"],
)

py_proto_library(
    name = "example_gen_proto_py_pb2",
    api_version = 2,
    deps = [":example_gen_proto"],
)

proto_library(
    name = "infra_validator_proto",
    srcs = ["infra_validator.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "infra_validator_py_pb2",
    api_version = 2,
    deps = [":infra_validator_proto"],
)

proto_library(
    name = "pusher_proto",
    srcs = ["pusher.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "pusher_proto_py_pb2",
    api_version = 2,
    deps = [":pusher_proto"],
)

proto_library(
    name = "trainer_proto",
    srcs = ["trainer.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "trainer_proto_py_pb2",
    api_version = 2,
    deps = [":trainer_proto"],
)

proto_library(
    name = "transform_proto",
    srcs = ["transform.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "transform_proto_py_pb2",
    api_version = 2,
    deps = [":transform_proto"],
)

proto_library(
    name = "tuner_proto",
    srcs = ["tuner.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "tuner_proto_py_pb2",
    api_version = 2,
    deps = [":tuner_proto"],
)
