"""Small hand-built ONNX graphs, built via onnx.helper, for frontend unit
tests that must not depend on the real MNIST model's external weights.
"""

import numpy as np
import onnx
import onnx.helper as oh

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
