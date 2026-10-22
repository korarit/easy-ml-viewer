# backend/scripts/generate_sample_onnx_deep.py
"""Phase 39 sample bootstrapper — ONNX deep tabular nets (pure ``onnx.helper``).

Builds small reproducible ``Gemm -> Relu -> Gemm -> Relu -> Gemm`` graphs
(no torch needed, 100% offline) plus matching CSVs:

- ``sample_deep_mlp_onnx.onnx`` / ``sample_deep_mlp_onnx.csv``
  (4-feature binary classifier, dynamic batch ``[None, 4]`` -> logits ``[None, 2]``)
- ``sample_deep_static_onnx.onnx`` / ``sample_deep_static_onnx.csv``
  (same topology but PyTorch-style static batch ``[1, 4]``)
- ``sample_deep_reg_onnx.onnx`` / ``sample_deep_reg_onnx.csv``
  (3-feature regressor, dynamic batch ``[None, 3]`` -> value ``[None, 1]``)

Run:  ``python scripts/generate_sample_onnx_deep.py [--out DIR]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_mlp(
    n_features: int,
    hidden: list[int],
    n_outputs: int,
    batch_dim: int | str | None,
    seed: int,
):  # type: ignore[no-untyped-def]
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    inits = []
    nodes = []
    prev = "input"
    width = int(n_features)
    for i, h in enumerate(hidden):
        w = (rng.normal(0.0, 0.6, size=(width, h)) / np.sqrt(max(width, 1))).astype(np.float32)
        b = np.zeros((h,), dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"W{i}"))
        inits.append(numpy_helper.from_array(b, f"B{i}"))
        h_name = f"h{i}"
        r_name = f"r{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"], [h_name]))
        nodes.append(helper.make_node("Relu", [h_name], [r_name]))
        prev = r_name
        width = int(h)
    w = (rng.normal(0.0, 0.6, size=(width, n_outputs)) / np.sqrt(max(width, 1))).astype(np.float32)
    b = np.zeros((n_outputs,), dtype=np.float32)
    inits.append(numpy_helper.from_array(w, "Wout"))
    inits.append(numpy_helper.from_array(b, "Bout"))
    nodes.append(helper.make_node("Gemm", [prev, "Wout", "Bout"], ["logits"]))
    batch = batch_dim if batch_dim is not None else None
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, n_features])
    Y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, n_outputs])
    graph = helper.make_graph(nodes, "tabular_dnn", [X], [Y], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def generate_classifier(output_dir: Path, static: bool = False) -> None:
    rng = np.random.default_rng(39)
    n = 500
    vibration = rng.uniform(0.0, 120.0, size=n)
    temp = rng.uniform(20.0, 110.0, size=n)
    pressure = rng.uniform(0.5, 10.0, size=n)
    rpm = rng.uniform(500.0, 3600.0, size=n)
    df = pd.DataFrame({"vibration_hz": vibration, "temp_c": temp, "pressure_bar": pressure, "rpm": rpm})
    Xn = np.column_stack([
        (df["vibration_hz"].to_numpy() - 60.0) / 35.0,
        (df["temp_c"].to_numpy() - 65.0) / 26.0,
        (df["pressure_bar"].to_numpy() - 5.0) / 2.8,
        (df["rpm"].to_numpy() - 2000.0) / 900.0,
    ])
    model = _build_mlp(4, [16, 8], 2, (1 if static else None), seed=39)
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    Xf = Xn.astype(np.float32)
    if static:
        # Static batch_size=1 export: unroll one row at a time (§2.3).
        parts = [sess.run(["logits"], {"input": Xf[i : i + 1]})[0] for i in range(Xf.shape[0])]
        logits = np.concatenate(parts, axis=0)
    else:
        logits = sess.run(["logits"], {"input": Xf})[0]
    proba = _softmax(np.asarray(logits))
    df["anomaly"] = (np.asarray(proba)[:, 1] > 0.5).astype(int)
    stem = "sample_deep_static_onnx" if static else "sample_deep_mlp_onnx"
    df.to_csv(output_dir / f"{stem}.csv", index=False, encoding="utf-8")
    (output_dir / f"{stem}.onnx").write_bytes(model.SerializeToString())


def generate_regressor(output_dir: Path) -> None:
    rng = np.random.default_rng(41)
    n = 500
    size_m2 = rng.uniform(30.0, 250.0, size=n)
    rooms = rng.uniform(1.0, 6.0, size=n)
    age = rng.uniform(0.0, 50.0, size=n)
    df = pd.DataFrame({"size_m2": size_m2, "rooms": rooms, "age_yr": age})
    Xn = np.column_stack([
        (df["size_m2"].to_numpy() - 140.0) / 63.0,
        (df["rooms"].to_numpy() - 3.5) / 1.4,
        (df["age_yr"].to_numpy() - 25.0) / 14.5,
    ])
    model = _build_mlp(3, [12, 8], 1, None, seed=41)
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    raw = np.asarray(sess.run(["logits"], {"input": Xn.astype(np.float32)})[0]).ravel()
    df["price"] = (raw * 50000.0 + 250000.0).astype(np.float64)
    df.to_csv(output_dir / "sample_deep_reg_onnx.csv", index=False, encoding="utf-8")
    (output_dir / "sample_deep_reg_onnx.onnx").write_bytes(model.SerializeToString())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 39 ONNX deep samples")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent.parent / "samples"),
        help="Output directory (default: repo samples/)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generate_classifier(out, static=False)
    generate_classifier(out, static=True)
    generate_regressor(out)
    print("wrote onnx deep mlp/static/regressor samples ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
