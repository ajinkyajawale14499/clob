"""Training entry point: LOBSTER -> features -> labels -> LightGBM -> ONNX.

Run via: `uv run python -m model.train` (no args; uses all 5 LOBSTER stocks).
Or import `train_one(k)` for tests + downstream tooling.

Pipeline:
    1. For each of 5 LOBSTER stocks: load message+book, extract trades,
       build_feature_matrix, generate 3-class labels for each K in LABEL_K_GRID.
    2. Time-split per stock (70% train / 30% val — single-day data caveat in ADR 0009).
    3. Concatenate across all 5 stocks (pooled).
    4. Balanced under-sampling on the TRAIN set per Briola 2024 (5000/class/day-like).
       Validation stays sequential / all-data.
    5. Train LightGBM multiclass for each K; pick best by val multi_logloss.
    6. Compute per-stock AUC (one-vs-rest macro, Up vs Down only — skip Stable).
    7. Save model + metadata + microprice LUT to model/artifacts/ (gitignored).

All commits stay green: pre-flight smoke tests run on minimal data; full training
is gated behind `pytest.mark.data` and only fires when LOBSTER files are present.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from data.ingestion.lobster_message import load_lobster_messages
from data.ingestion.lobster_orderbook import (
    join_messages_orderbook,
    load_lobster_orderbook,
)
from data.ingestion.lobster_trades import extract_trades
from model.features import build_feature_matrix
from model.labels import class_share, make_labels_3class
from model.microprice_g import MicropriceLut, fit_microprice_g
from model.schema import (
    ALL_TICKERS,
    FEATURE_DTYPE,
    FEATURE_NAMES,
    LABEL_K_GRID,
    LIGHTGBM_EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    SCHEMA_VERSION,
)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


@dataclass
class PerStockMetrics:
    auc_up_vs_down: float       # one-vs-rest binary on non-stable rows
    logloss: float
    n_val: int
    class_shares_val: dict[int, float]


@dataclass
class TrainResult:
    k: int
    n_features: int
    n_trees_used: int
    n_rows_train: int
    n_rows_val: int
    val_logloss_pool: float
    val_auc_pool_up_vs_down: float
    per_stock: dict[str, PerStockMetrics]
    train_class_shares: dict[int, float]
    val_class_shares: dict[int, float]


def _ensure_microprice_lut(rebuild: bool = False) -> MicropriceLut:
    """Load or fit the Stoikov G(I,S) LUT. Cached at artifacts/microprice_g.json."""
    lut_path = ARTIFACTS_DIR / "microprice_g.json"
    if lut_path.exists() and not rebuild:
        return MicropriceLut.load(lut_path)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for ticker in ALL_TICKERS:
        msg = load_lobster_messages(
            next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
        book = load_lobster_orderbook(
            next(SAMPLE_DIR.glob(f"{ticker}_*_orderbook_*.csv")), n_levels=10)
        joined = join_messages_orderbook(msg, book)
        frames.append(
            joined.select("bid_price_l1", "ask_price_l1", "bid_size_l1", "ask_size_l1")
            .head(50_000)
        )
    pooled = pl.concat(frames)
    lut = fit_microprice_g(pooled, n_imbalance_buckets=11, n_spread_buckets=5,
                            tick_size=100)
    lut.save(lut_path)
    return lut


def _load_stock_features_and_labels(
    ticker: str, k: int, lut: MicropriceLut
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, ts_ns) aligned by row index, dropping the trailing K rows."""
    msg = load_lobster_messages(
        next(SAMPLE_DIR.glob(f"{ticker}_*_message_*.csv")))
    book = load_lobster_orderbook(
        next(SAMPLE_DIR.glob(f"{ticker}_*_orderbook_*.csv")), n_levels=10)
    joined = join_messages_orderbook(msg, book)
    trades = extract_trades(msg)

    feats = build_feature_matrix(joined, trades, ticker=ticker, lut=lut)
    labelled = make_labels_3class(joined, k_events=k)

    n = min(feats.height, labelled.height)
    X = feats.head(n).select(FEATURE_NAMES).to_numpy().astype(FEATURE_DTYPE)
    y = labelled.head(n)["label"].to_numpy().astype(np.int8)
    ts = feats.head(n)["ts_ns"].to_numpy()
    return X, y, ts


def _time_split(
    X: np.ndarray, y: np.ndarray, ts: np.ndarray, train_frac: float = 0.7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(len(ts) * train_frac)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _balanced_undersample(
    X: np.ndarray, y: np.ndarray, *, samples_per_class: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Briola 2024-style balanced under-sampling on the training set.

    Picks `samples_per_class` random rows per label class. Caps at the
    minimum class population so we don't sample with replacement.
    """
    rng = np.random.default_rng(seed)
    out_X, out_y = [], []
    for cls in (0, 1, 2):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        take = min(samples_per_class, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        out_X.append(X[chosen])
        out_y.append(y[chosen])
    X_bal = np.concatenate(out_X, axis=0)
    y_bal = np.concatenate(out_y, axis=0)
    # Shuffle.
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def _auc_up_vs_down(y_true: np.ndarray, p_classes: np.ndarray) -> float:
    """One-vs-rest binary AUC: P(Up) vs P(Down), excluding Stable rows.

    Literature-equivalent metric (Briola 2024 reports MCC on a similar setup).
    Reflects the model's ability to distinguish actual directional moves,
    conditional on having a directional move.

    Returns NaN when the non-stable sample is too small to be reliable.
    Common at small K for large-tick stocks (INTC/MSFT at K=10 have ~99.9%
    stable rows; AUC computed on the tiny remainder is a sampling artifact,
    not real signal). For those stocks, prefer K=50 or K=100.
    """
    mask = (y_true == 0) | (y_true == 2)
    # Require at least 1000 non-stable rows for a meaningful AUC.
    if mask.sum() < 1000:
        return float("nan")
    y_bin = (y_true[mask] == 2).astype(int)  # 1 = Up, 0 = Down
    # P(Up) - P(Down): higher means more confident Up.
    score = p_classes[mask, 2] - p_classes[mask, 0]
    return float(roc_auc_score(y_bin, score))


def train_one(k: int, *, seed: int = 42, samples_per_class: int = 5000,
              verbose: bool = False) -> tuple[lgb.LGBMClassifier, TrainResult]:
    """Train one LightGBM model for label horizon K. Returns (model, metrics)."""
    lut = _ensure_microprice_lut()

    per_stock_data = []
    for ticker in ALL_TICKERS:
        X, y, ts = _load_stock_features_and_labels(ticker, k, lut)
        Xtr, ytr, Xva, yva = _time_split(X, y, ts)
        per_stock_data.append((ticker, Xtr, ytr, Xva, yva))

    # Pooled training + per-stock validation.
    Xtr_pool = np.concatenate([t[1] for t in per_stock_data])
    ytr_pool = np.concatenate([t[2] for t in per_stock_data])
    Xva_pool = np.concatenate([t[3] for t in per_stock_data])
    yva_pool = np.concatenate([t[4] for t in per_stock_data])

    # Balanced under-sampling on TRAIN only (val stays sequential per Briola 2024).
    n_per_class_per_stock = samples_per_class // len(ALL_TICKERS) or 1000
    total_per_class = n_per_class_per_stock * len(ALL_TICKERS)
    Xtr_bal, ytr_bal = _balanced_undersample(
        Xtr_pool, ytr_pool, samples_per_class=total_per_class, seed=seed)

    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS, random_state=seed)
    callbacks = [lgb.early_stopping(LIGHTGBM_EARLY_STOPPING_ROUNDS,
                                     first_metric_only=True, verbose=verbose)]
    if not verbose:
        callbacks.append(lgb.log_evaluation(0))
    model.fit(
        Xtr_bal, ytr_bal,
        eval_set=[(Xva_pool, yva_pool)],
        callbacks=callbacks,
    )

    # Pooled val metrics.
    p_val_pool = model.predict_proba(Xva_pool)
    val_ll = float(log_loss(yva_pool, p_val_pool, labels=[0, 1, 2]))
    val_auc = _auc_up_vs_down(yva_pool, p_val_pool)

    # Per-stock val metrics.
    per_stock: dict[str, PerStockMetrics] = {}
    for ticker, _, _, Xva, yva in per_stock_data:
        p_va = model.predict_proba(Xva)
        per_stock[ticker] = PerStockMetrics(
            auc_up_vs_down=_auc_up_vs_down(yva, p_va),
            logloss=float(log_loss(yva, p_va, labels=[0, 1, 2])),
            n_val=len(yva),
            class_shares_val=class_share(pl.Series("label", yva, dtype=pl.Int8)),
        )

    result = TrainResult(
        k=k,
        n_features=len(FEATURE_NAMES),
        n_trees_used=int(model.best_iteration_ or model.n_estimators),
        n_rows_train=len(ytr_bal),
        n_rows_val=len(yva_pool),
        val_logloss_pool=val_ll,
        val_auc_pool_up_vs_down=val_auc,
        per_stock=per_stock,
        train_class_shares=class_share(pl.Series("label", ytr_bal, dtype=pl.Int8)),
        val_class_shares=class_share(pl.Series("label", yva_pool, dtype=pl.Int8)),
    )
    return model, result


def grid_search_k(verbose: bool = False) -> tuple[lgb.LGBMClassifier, TrainResult, dict]:
    """Train one model per K; pick best by val multi_logloss (smaller is better)."""
    best_model = None
    best_result: TrainResult | None = None
    grid: dict[int, dict] = {}
    for k in LABEL_K_GRID:
        if verbose:
            print(f"\n=== Training K={k} ===")
        m, r = train_one(k, verbose=verbose)
        grid[k] = {
            "val_logloss_pool": r.val_logloss_pool,
            "val_auc_pool_up_vs_down": r.val_auc_pool_up_vs_down,
            "n_trees_used": r.n_trees_used,
        }
        if best_result is None or r.val_logloss_pool < best_result.val_logloss_pool:
            best_model = m
            best_result = r
    return best_model, best_result, grid


def export_to_onnx(model: lgb.LGBMClassifier,
                   output_path: Path = ARTIFACTS_DIR / "model.onnx") -> Path:
    """Export trained LightGBM to ONNX.

    Per ADR 0006: target_opset=15 (onnxmltools 1.16 LightGBM converter cap;
    higher opsets fail at conversion). Uses onnxmltools's FloatTensorType
    (NOT onnxconverter_common's — the shape calculator does an isinstance check).

    Validate the export with `validate_onnx_drift(model, output_path)` before
    trusting it on the C++ hot path.
    """
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert.lightgbm.convert import convert

    output_path.parent.mkdir(parents=True, exist_ok=True)
    initial_types = [("input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert(
        model.booster_,
        initial_types=initial_types,
        name=f"clob_v1.0_schema_v{SCHEMA_VERSION}",
        target_opset=15,
    )
    output_path.write_bytes(onnx_model.SerializeToString())
    return output_path


def validate_onnx_drift(model: lgb.LGBMClassifier, onnx_path: Path,
                       n_samples: int = 1000, seed: int = 42) -> dict:
    """Sweep n_samples random feature vectors through both LightGBM (float64
    internal) and onnxruntime (float32 internal); return drift statistics.

    Per ADR 0006: expected `max_abs_diff` < 1e-4 and `rmse` << 1e-4 when
    n_estimators <= 300. If drift exceeds the gate, the train/serve skew test
    (W10 task 6.5) will fire.
    """
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, len(FEATURE_NAMES))).astype(FEATURE_DTYPE)

    # LightGBM (float64 internal).
    p_lgb = model.predict_proba(X).astype(np.float64)

    # onnxruntime (float32 internal).
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = sess.run(None, {"input": X})
    # LightGBM ONNX outputs: [label, probabilities_zipmap]. probabilities is a
    # list of dicts {class_id: prob}. Build the array.
    probs_list = raw[1]
    p_onnx = np.array(
        [[d[0], d[1], d[2]] for d in probs_list], dtype=np.float64
    )

    diff = np.abs(p_lgb - p_onnx)
    return {
        "n_samples": n_samples,
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def save_artifacts(model: lgb.LGBMClassifier, result: TrainResult, grid: dict) -> None:
    """Write model.lgb (native), model.onnx (deployment), model_meta.json."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Native LightGBM (for TreeLite W14 + reproducibility).
    model.booster_.save_model(str(ARTIFACTS_DIR / "model.lgb"))

    # ONNX export for the C++ hot path (ADR 0006).
    onnx_path = export_to_onnx(model)
    drift = validate_onnx_drift(model, onnx_path)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_names": FEATURE_NAMES,
        "n_features": result.n_features,
        "k_best": result.k,
        "k_grid": grid,
        "n_trees_used": result.n_trees_used,
        "n_rows_train": result.n_rows_train,
        "n_rows_val": result.n_rows_val,
        "val_logloss_pool": result.val_logloss_pool,
        "val_auc_pool_up_vs_down": result.val_auc_pool_up_vs_down,
        "train_class_shares": result.train_class_shares,
        "val_class_shares": result.val_class_shares,
        "per_stock": {t: asdict(m) for t, m in result.per_stock.items()},
        "lightgbm_params": LIGHTGBM_PARAMS,
        "early_stopping_rounds": LIGHTGBM_EARLY_STOPPING_ROUNDS,
        "onnx_drift": drift,
    }
    (ARTIFACTS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))


def main(verbose: bool = True) -> None:
    print(f"Sample dir: {SAMPLE_DIR}")
    print(f"Artifacts:  {ARTIFACTS_DIR}")
    print(f"K grid:     {LABEL_K_GRID}")
    model, result, grid = grid_search_k(verbose=verbose)
    save_artifacts(model, result, grid)
    print(f"\n=== Best K={result.k} ===")
    print(f"  val_logloss_pool       = {result.val_logloss_pool:.4f}")
    print(f"  val_auc_pool_up_vs_down = {result.val_auc_pool_up_vs_down:.4f}")
    print(f"  n_trees_used           = {result.n_trees_used}")
    print("  Per-stock AUC (up vs down):")
    for ticker, m in result.per_stock.items():
        print(f"    {ticker}: AUC={m.auc_up_vs_down:.4f}, logloss={m.logloss:.4f}, "
              f"shares={m.class_shares_val}")
    print(f"\nSaved: model.lgb + model_meta.json -> {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
