"""Phase 13 — counterfactual finder + what-if mutation domain services.

Implements the spec §4.1 ``CounterfactualFinderService`` (empirical KDTree
nearest-opposite + 12-step binary line search toward the decision boundary)
with the §5.1 / §7 guards:

- reference pool subsampled to ``<= 5_000`` rows (bounded RAM / latency),
- immutable ``locked_feature_indices`` projected back onto every candidate
  and re-validated through ``predict_fn``,
- synthetic uniform-grid fallback (500 points) when the CSV holds a single
  predicted class (edge #6),
- integer / binary columns rounded on every bisection step (edge #2),
- all math in ``float32``; zero-``std`` guard (``std < 1e-9 -> 1.0``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:  # pragma: no cover - sklearn always pulls scipy in
    KDTree = None  # type: ignore[assignment]

#: Hard cap on the KDTree reference pool (§5.1: <= 5_000 rows, ~4 MB).
MAX_REFERENCE_ROWS = 5_000
#: Rows above which the pool is stride-subsampled before the TF search.
SUBSAMPLE_ABOVE_ROWS = 20_000
#: Bisection iterations pulling the candidate toward the boundary.
BISECTION_STEPS = 12
#: Synthetic fallback grid size when no opposite-class row exists (edge #6).
SYNTHETIC_GRID_POINTS = 500


def detect_feature_kinds(X: np.ndarray) -> list[str]:
    """Classify each column as ``binary`` | ``integer`` | ``float`` (edge #2)."""
    mat = np.asarray(X, dtype=np.float32)
    kinds: list[str] = []
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            kinds.append("float")
            continue
        uniq = np.unique(finite)
        if uniq.size <= 2 and bool(np.all((uniq == 0.0) | (uniq == 1.0))):
            kinds.append("binary")
            continue
        if bool(np.all(np.abs(finite - np.round(finite)) < 1e-6)):
            kinds.append("integer")
            continue
        kinds.append("float")
    return kinds


def round_int_binary_columns(vec: np.ndarray, kinds: list[str]) -> np.ndarray:
    """Snap integer/binary columns to whole numbers (edge #2)."""
    out = np.asarray(vec, dtype=np.float32).copy()
    for j, kind in enumerate(kinds):
        if j >= out.shape[0]:
            break
        if kind in ("integer", "binary"):
            out[j] = float(np.round(float(out[j])))
            if kind == "binary":
                out[j] = float(np.clip(out[j], 0.0, 1.0))
    return out.astype(np.float32)


def subsample_reference_pool(
    X: np.ndarray,
    y_preds: np.ndarray,
    max_rows: int = MAX_REFERENCE_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stride subsample keeping class balance (§5.1)."""
    n = int(X.shape[0])
    if n <= max_rows:
        return X, y_preds
    # Stratified-ish: stride sample per predicted class, then interleave.
    classes = np.unique(y_preds)
    picked: list[np.ndarray] = []
    per_class = max(1, max_rows // max(1, len(classes)))
    for c in classes:
        idx = np.flatnonzero(y_preds == c)
        if idx.size > per_class:
            stride = max(1, idx.size // per_class)
            idx = idx[::stride][:per_class]
        picked.append(idx)
    order = np.concatenate(picked) if picked else np.empty((0,), dtype=np.int64)
    if order.size > max_rows:
        order = order[:max_rows]
    order = np.sort(order)
    return (
        np.ascontiguousarray(X[order], dtype=np.float32),
        np.ascontiguousarray(y_preds[order]),
    )


class CounterfactualFinderService:
    """Minimum-change counterfactual search honoring immutable locks."""

    @staticmethod
    def find_minimum_change_counterfactual(
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_dataset: np.ndarray,
        y_predictions: np.ndarray,
        base_row: np.ndarray,
        locked_feature_indices: list[int],
        target_class_or_threshold: float,
        is_classification: bool = True,
        feature_kinds: list[str] | None = None,
    ) -> tuple[bool, np.ndarray, str]:
        n_samples, n_features = X_dataset.shape
        base = np.asarray(base_row, dtype=np.float32).ravel().copy()
        locked = sorted({int(i) for i in locked_feature_indices if 0 <= int(i) < n_features})
        kinds = list(feature_kinds) if feature_kinds else detect_feature_kinds(X_dataset)

        if len(locked) >= n_features and n_features > 0:
            return False, base.copy(), "กรุณาปลดล็อกอย่างน้อย 1 Feature เพื่อค้นหา Counterfactual"

        feature_stds = np.nanstd(X_dataset.astype(np.float64, copy=False), axis=0)
        feature_stds = np.where(
            np.isfinite(feature_stds) & (feature_stds >= 1e-9), feature_stds, 1.0
        )

        # Pool may be huge: stride-subsample for the candidate search only.
        if n_samples > SUBSAMPLE_ABOVE_ROWS:
            pool_X, pool_y = subsample_reference_pool(
                np.asarray(X_dataset, dtype=np.float32), np.asarray(y_predictions).ravel()
            )
        else:
            pool_X = np.asarray(X_dataset, dtype=np.float32)
            pool_y = np.asarray(y_predictions).ravel()

        # 1. Candidate pool: rows already predicting the target.
        if is_classification:
            target_cls = int(round(float(target_class_or_threshold)))
            candidate_mask = np.round(pool_y) == target_cls
        else:
            try:
                base_pred = float(predict_fn(base.reshape(1, -1))[0])
            except (TypeError, ValueError, IndexError):
                base_pred = float(np.mean(pool_y)) if pool_y.size else 0.0
            if float(target_class_or_threshold) >= base_pred:
                candidate_mask = pool_y >= float(target_class_or_threshold)
            else:
                candidate_mask = pool_y <= float(target_class_or_threshold)

        candidates_X = pool_X[candidate_mask] if bool(np.any(candidate_mask)) else np.empty(
            (0, n_features), dtype=np.float32
        )

        # 2. Project locked columns back onto the base values.
        def _project(cands: np.ndarray) -> np.ndarray:
            proj = cands.astype(np.float32, copy=True)
            for idx in locked:
                proj[:, idx] = base[idx]
            return round_int_binary_columns_matrix(proj, kinds)

        # 3. Re-validate through the model (locks can flip predictions back).
        def _valid_mask(cands: np.ndarray) -> np.ndarray:
            if len(cands) == 0:
                return np.zeros((0,), dtype=bool)
            preds = np.asarray(predict_fn(np.ascontiguousarray(cands, dtype=np.float32))).ravel()
            if is_classification:
                return np.round(preds) == round(float(target_class_or_threshold))
            try:
                base_pred_local = float(predict_fn(base.reshape(1, -1))[0])
            except (TypeError, ValueError, IndexError):
                base_pred_local = float(np.mean(pool_y)) if pool_y.size else 0.0
            if float(target_class_or_threshold) >= base_pred_local:
                return preds >= float(target_class_or_threshold)
            return preds <= float(target_class_or_threshold)

        valid_candidates = np.empty((0, n_features), dtype=np.float32)
        if len(candidates_X):
            projected = _project(candidates_X)
            mask = _valid_mask(projected)
            valid_candidates = projected[mask] if bool(np.any(mask)) else valid_candidates

        # Edge #6: single-class CSV -> synthetic uniform grid inside [min, max].
        if len(valid_candidates) == 0:
            synth = CounterfactualFinderService._synthetic_grid(
                pool_X if len(pool_X) else np.asarray(X_dataset, dtype=np.float32),
                base,
                locked,
                kinds,
            )
            if len(synth):
                mask = _valid_mask(synth)
                valid_candidates = synth[mask] if bool(np.any(mask)) else valid_candidates

        if len(valid_candidates) == 0:
            if locked:
                return (
                    False,
                    base.copy(),
                    "ไม่สามารถเปลี่ยนคำตัดสินได้เนื่องจากคอลัมน์สำคัญถูกล็อกไว้ (Immutable Lock)",
                )
            return False, base.copy(), "ไม่พบตัวอย่างคลาสเป้าหมายในชุดข้อมูลอ้างอิง"

        # 4. Normalized Euclidean KDTree nearest neighbour.
        if KDTree is None:  # pragma: no cover - scipy always present via sklearn
            dists = np.linalg.norm((valid_candidates - base) / feature_stds, axis=1)
            best_candidate = valid_candidates[int(np.argmin(dists))].copy()
        else:
            norm_candidates = (valid_candidates / feature_stds).astype(np.float32)
            norm_base = (base / feature_stds).reshape(1, -1).astype(np.float32)
            norm_candidates = np.nan_to_num(
                norm_candidates, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            norm_base = np.nan_to_num(norm_base, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            tree = KDTree(norm_candidates)
            _, nearest_idx = tree.query(norm_base, k=1)
            best_candidate = valid_candidates[int(np.atleast_1d(nearest_idx)[0])].copy()

        # 5. Binary line search toward the boundary (12 iterations).
        low_vec = base.copy().astype(np.float32)
        high_vec = best_candidate.copy().astype(np.float32)
        for _ in range(BISECTION_STEPS):
            mid_vec = (0.5 * (low_vec.astype(np.float64) + high_vec.astype(np.float64))).astype(
                np.float32
            )
            mid_vec = round_int_binary_columns(mid_vec, kinds)
            for idx in locked:
                mid_vec[idx] = base[idx]
            try:
                mid_pred = float(predict_fn(mid_vec.reshape(1, -1))[0])
            except (TypeError, ValueError, IndexError):
                break
            if is_classification:
                is_flipped = int(round(mid_pred)) == int(round(float(target_class_or_threshold)))
            else:
                try:
                    base_pred_local = float(predict_fn(base.reshape(1, -1))[0])
                except (TypeError, ValueError, IndexError):
                    base_pred_local = float(mid_pred)
                is_flipped = (
                    mid_pred >= float(target_class_or_threshold)
                    if float(target_class_or_threshold) >= base_pred_local
                    else mid_pred <= float(target_class_or_threshold)
                )
            if is_flipped:
                high_vec = mid_vec
            else:
                low_vec = mid_vec

        high_vec = round_int_binary_columns(high_vec, kinds)
        for idx in locked:
            high_vec[idx] = base[idx]
        return True, high_vec.astype(np.float32), "ค้นหาจุด Counterfactual ที่ใช้การปรับค่าน้อยที่สุดสำเร็จ"

    @staticmethod
    def _synthetic_grid(
        reference_X: np.ndarray,
        base_row: np.ndarray,
        locked: list[int],
        kinds: list[str],
        n_points: int = SYNTHETIC_GRID_POINTS,
    ) -> np.ndarray:
        """Uniform random grid inside per-column [min, max] (edge #6)."""
        ref = np.asarray(reference_X, dtype=np.float32)
        if ref.size == 0:
            return np.empty((0, base_row.size), dtype=np.float32)
        finite = np.where(np.isfinite(ref), ref, 0.0)
        col_min = finite.min(axis=0).astype(np.float32)
        col_max = finite.max(axis=0).astype(np.float32)
        span = col_max - col_min
        span[~np.isfinite(span) | (span < 1e-9)] = 1.0
        rng = np.random.default_rng(42)
        grid = rng.random((n_points, ref.shape[1]), dtype=np.float32) * span + col_min
        grid = round_int_binary_columns_matrix(grid.astype(np.float32), kinds)
        for idx in locked:
            if 0 <= idx < grid.shape[1]:
                grid[:, idx] = base_row[idx]
        return np.ascontiguousarray(grid, dtype=np.float32)


def round_int_binary_columns_matrix(mat: np.ndarray, kinds: list[str]) -> np.ndarray:
    """Matrix variant of :func:`round_int_binary_columns`."""
    out = np.asarray(mat, dtype=np.float32).copy()
    for j, kind in enumerate(kinds):
        if j >= out.shape[1]:
            break
        if kind in ("integer", "binary"):
            out[:, j] = np.round(out[:, j])
            if kind == "binary":
                out[:, j] = np.clip(out[:, j], 0.0, 1.0)
    return out.astype(np.float32)
