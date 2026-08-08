"""Small hand-built ONNX graphs, built via onnx.helper (or, for the one
deliberately-invalid fixture, directly via onnx_ir), for frontend and
structural-pass unit tests that must not depend on the real MNIST model's
external weights.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx_ir as ir

OPSET = 18


def _save(model: onnx.ModelProto, path) -> str:
    onnx.save(model, str(path))
    return str(path)


def conv_relu_graph(tmp_path) -> str:
    """Minimal 2-node Conv -> Relu graph, fully static shapes."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 2, 2])

    weight = oh.make_tensor(
        "w", onnx.TensorProto.FLOAT, [1, 1, 3, 3], np.ones(9, dtype=np.float32)
    )
    bias = oh.make_tensor("b", onnx.TensorProto.FLOAT, [1], np.zeros(1, dtype=np.float32))

    conv = oh.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    relu = oh.make_node("Relu", inputs=["conv_out"], outputs=["y"])

    graph = oh.make_graph(
        [conv, relu], "conv_relu", [x], [y], initializer=[weight, bias]
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "conv_relu.onnx")


def unsupported_op_graph(tmp_path) -> str:
    """Minimal graph containing an op outside SUPPORTED_OPS (Sigmoid)."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])

    node = oh.make_node("Sigmoid", inputs=["x"], outputs=["y"])
    graph = oh.make_graph([node], "unsupported_op", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "unsupported_op.onnx")


def dynamic_batch_graph(tmp_path) -> str:
    """Minimal graph with a symbolic (dynamic) batch dim on the input."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["batch", 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["batch", 4])

    node = oh.make_node("Relu", inputs=["x"], outputs=["y"])
    graph = oh.make_graph([node], "dynamic_batch", [x], [y])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "dynamic_batch.onnx")


def div_relu_graph(tmp_path) -> str:
    """Minimal Div(input, scalar) -> Relu graph, for fold_div_into_quant."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])

    scale = oh.make_tensor("scale", onnx.TensorProto.FLOAT, [], [2.0])

    div = oh.make_node("Div", inputs=["x", "scale"], outputs=["div_out"])
    relu = oh.make_node("Relu", inputs=["div_out"], outputs=["y"])

    graph = oh.make_graph([div, relu], "div_relu", [x], [y], initializer=[scale])
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "div_relu.onnx")


def conv_relu_branching_graph(tmp_path) -> str:
    """Conv output feeds *two* Relu consumers, so fuse_relu's sole-consumer
    guard fires (branching input this compiler doesn't support yet)."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    y1 = oh.make_tensor_value_info("y1", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    y2 = oh.make_tensor_value_info("y2", onnx.TensorProto.FLOAT, [1, 1, 2, 2])

    weight = oh.make_tensor(
        "w", onnx.TensorProto.FLOAT, [1, 1, 3, 3], np.ones(9, dtype=np.float32)
    )
    bias = oh.make_tensor("b", onnx.TensorProto.FLOAT, [1], np.zeros(1, dtype=np.float32))

    conv = oh.make_node(
        "Conv",
        inputs=["x", "w", "b"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    relu1 = oh.make_node("Relu", inputs=["conv_out"], outputs=["y1"])
    relu2 = oh.make_node("Relu", inputs=["conv_out"], outputs=["y2"])

    graph = oh.make_graph(
        [conv, relu1, relu2], "conv_relu_branching", [x], [y1, y2], initializer=[weight, bias]
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "conv_relu_branching.onnx")


def conv_reshape_gemm_graph(tmp_path) -> str:
    """Conv -> Reshape(flatten) -> Gemm, for drop_noop_reshape's happy path."""
    x = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    y = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])

    conv_weight = oh.make_tensor(
        "cw", onnx.TensorProto.FLOAT, [1, 1, 3, 3], np.ones(9, dtype=np.float32)
    )
    conv_bias = oh.make_tensor("cb", onnx.TensorProto.FLOAT, [1], np.zeros(1, dtype=np.float32))
    reshape_shape = oh.make_tensor("shape", onnx.TensorProto.INT64, [2], [1, 4])
    gemm_weight = oh.make_tensor(
        "gw", onnx.TensorProto.FLOAT, [2, 4], np.ones(8, dtype=np.float32)
    )
    gemm_bias = oh.make_tensor("gb", onnx.TensorProto.FLOAT, [2], np.zeros(2, dtype=np.float32))

    conv = oh.make_node(
        "Conv",
        inputs=["x", "cw", "cb"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )
    reshape = oh.make_node("Reshape", inputs=["conv_out", "shape"], outputs=["view"])
    gemm = oh.make_node("Gemm", inputs=["view", "gw", "gb"], outputs=["y"], transB=1)

    graph = oh.make_graph(
        [conv, reshape, gemm],
        "conv_reshape_gemm",
        [x],
        [y],
        initializer=[conv_weight, conv_bias, reshape_shape, gemm_weight, gemm_bias],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return _save(model, tmp_path / "conv_reshape_gemm.onnx")


def reshape_bad_shape_graph() -> ir.Model:
    """A Reshape whose declared output has a *different* total element
    count than its input (4 -> 3) — invalid under the ONNX spec, so it
    can't be built via onnx.helper + onnx.checker; built directly with
    onnx_ir instead, bypassing the checker entirely, to exercise
    DropNoopReshapePass's defensive element-count assertion.
    """
    x = ir.Value(name="x", shape=ir.Shape([1, 4]), type=ir.TensorType(ir.DataType.FLOAT))
    shape_data = ir.tensor(np.array([1, 3], dtype=np.int64), name="bad_shape")
    shape_value = ir.Value(
        name="bad_shape",
        shape=ir.Shape([2]),
        type=ir.TensorType(ir.DataType.INT64),
        const_value=shape_data,
    )
    y = ir.Value(name="y", shape=ir.Shape([1, 3]), type=ir.TensorType(ir.DataType.FLOAT))

    node = ir.node("Reshape", inputs=[x, shape_value], outputs=[y], name="node_reshape")

    graph = ir.Graph(
        inputs=[x],
        outputs=[y],
        nodes=[node],
        initializers=[shape_value],
        opset_imports={"": OPSET},
        name="bad_reshape",
    )
    return ir.Model(graph, ir_version=9)
