"""ONNX graph validation — structural integrity of the exported model."""

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest

from model.schema import FEATURE_NAMES

pytestmark = pytest.mark.data

MODEL_PATH = Path(__file__).parents[2] / "model" / "artifacts" / "model.onnx"


def _have_model() -> bool:
    return MODEL_PATH.exists()


if not _have_model():
    pytest.skip("model/artifacts/model.onnx missing", allow_module_level=True)


def test_onnx_checker_passes():
    """onnx.checker.check_model validates the protobuf + opset compatibility."""
    m = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(m)  # raises ValidationError on issues


def test_onnx_opset_at_or_below_15():
    """ADR 0006: onnxmltools 1.16 LightGBM converter caps at opset 15."""
    m = onnx.load(str(MODEL_PATH))
    for opset in m.opset_import:
        # Default domain ("") opset is the one that gates the converter.
        if opset.domain == "":
            assert opset.version <= 15, \
                f"default opset {opset.version} > 15 (ADR 0006 cap)"


def test_onnx_input_shape_matches_schema():
    sess = ort.InferenceSession(str(MODEL_PATH))
    inp = sess.get_inputs()[0]
    # Shape: [None (batch), 19 (features)]
    assert len(inp.shape) == 2
    assert inp.shape[1] == len(FEATURE_NAMES)
    assert inp.type == "tensor(float)"


def test_onnx_outputs_label_and_probabilities():
    """zipmap=False export -> outputs are [label (int64), probabilities (float)]."""
    sess = ort.InferenceSession(str(MODEL_PATH))
    outs = sess.get_outputs()
    assert len(outs) == 2
    # Output 1 should be a plain (N, 3) float tensor — not a ZipMap.
    assert outs[1].type == "tensor(float)"
    # Shape: [None, 3]
    assert len(outs[1].shape) == 2
    assert outs[1].shape[1] == 3


def test_onnx_runs_on_cpu_provider():
    """All operators must be supported on CPUExecutionProvider (no CUDA-only ops)."""
    sess = ort.InferenceSession(str(MODEL_PATH),
                                  providers=["CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CPUExecutionProvider"
