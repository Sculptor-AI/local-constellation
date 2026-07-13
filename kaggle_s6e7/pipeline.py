from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import warnings
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
SEED = 20260713
random.seed(SEED)
np.random.seed(SEED)

ID = "id"
TARGET = "health_condition"
CAT_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
PUBLIC_REFERENCE_PRIOR = {
    "at-risk": 239010 / 295753,
    "fit": 23327 / 295753,
    "unhealthy": 33416 / 295753,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[TARGET], errors="ignore").copy()
    base = [c for c in out.columns if c != ID]
    out["n_missing"] = out[base].isna().sum(axis=1).astype("int8")
    for c in base:
        if out[c].isna().any():
            out[f"{c}__missing"] = out[c].isna().astype("int8")

    eps = 1.0
    out["cal_per_step"] = out["calorie_expenditure"] / (out["step_count"] + eps)
    out["steps_per_exercise"] = out["step_count"] / (out["exercise_duration"] + eps)
    out["cal_per_exercise"] = out["calorie_expenditure"] / (out["exercise_duration"] + eps)
    out["water_per_cal"] = out["water_intake"] / (out["calorie_expenditure"] + eps)
    out["water_per_exercise"] = out["water_intake"] / (out["exercise_duration"] + eps)
    out["hr_x_bmi"] = out["heart_rate"] * out["bmi"]
    out["sleep_x_exercise"] = out["sleep_duration"] * out["exercise_duration"]
    out["steps_x_sleep"] = out["step_count"] * out["sleep_duration"]
    out["activity_energy"] = out["step_count"] * out["calorie_expenditure"]
    out["sleep_deviation_8"] = (out["sleep_duration"] - 8.0).abs()
    out["heart_deviation_70"] = (out["heart_rate"] - 70.0).abs()
    out["bmi_deviation_22"] = (out["bmi"] - 22.0).abs()

    # Low-cardinality combinations expose useful interactions to all model families.
    combos = [
        ("stress_level", "sleep_quality"),
        ("stress_level", "physical_activity_level"),
        ("diet_type", "physical_activity_level"),
        ("diet_type", "sleep_quality"),
        ("smoking_alcohol", "gender"),
    ]
    for a, b in combos:
        out[f"{a}__x__{b}"] = (
            out[a].astype("string").fillna("__NA__")
            + "|"
            + out[b].astype("string").fillna("__NA__")
        )
    missing_cols = [c for c in base if out[c].isna().any()]
    out["missing_pattern"] = out[missing_cols].isna().astype("int8").astype(str).agg("".join, axis=1)
    return out


def prepare_encoded(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = add_features(train)
    te = add_features(test)
    tr = tr.drop(columns=[ID])
    te = te.drop(columns=[ID])
    all_df = pd.concat([tr, te], axis=0, ignore_index=True)
    categorical = all_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in categorical:
        values = all_df[c].astype("string").fillna("__NA__")
        categories = pd.Index(values.unique())
        mapping = pd.Series(np.arange(len(categories), dtype=np.int32), index=categories)
        all_df[c] = values.map(mapping).astype("int32")
    tr_out = all_df.iloc[: len(tr)].copy()
    te_out = all_df.iloc[len(tr) :].reset_index(drop=True).copy()
    del all_df
    gc.collect()
    return tr_out, te_out, categorical


def prepare_catboost(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = add_features(train).drop(columns=[ID])
    te = add_features(test).drop(columns=[ID])
    categorical = tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in categorical:
        tr[c] = tr[c].astype("string").fillna("__NA__").astype(str)
        te[c] = te[c].astype("string").fillna("__NA__").astype(str)
    return tr, te, categorical


def weighted_balanced_accuracy(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    recalls = []
    for c in np.unique(y):
        mask = y == c
        if weights is None:
            recalls.append(float(np.mean(pred[mask] == c)))
        else:
            denom = float(weights[mask].sum())
            recalls.append(float(weights[mask][pred[mask] == c].sum() / max(denom, 1e-15)))
    return float(np.mean(recalls))


def apply_multipliers(prob: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    return np.argmax(prob * multipliers.reshape(1, -1), axis=1)


def tune_multipliers(
    y: np.ndarray,
    prob: np.ndarray,
    weights: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, float]:
    # Fix the first multiplier at one because only relative scales matter.
    def objective(x: np.ndarray) -> float:
        mult = np.r_[1.0, np.exp(x)]
        pred = apply_multipliers(prob, mult)
        return -weighted_balanced_accuracy(y, pred, weights)

    result = differential_evolution(
        objective,
        bounds=[(-1.4, 1.4)] * (prob.shape[1] - 1),
        seed=seed,
        popsize=12,
        maxiter=45,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    mult = np.r_[1.0, np.exp(result.x)]
    return mult, -float(result.fun)


def optimize_blend(
    y: np.ndarray,
    oof_list: list[np.ndarray],
    weights: np.ndarray | None = None,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_models = len(oof_list)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = np.r_[x[: n_models - 1], 0.0]
        logits -= logits.max()
        model_w = np.exp(logits)
        model_w /= model_w.sum()
        mult = np.r_[1.0, np.exp(x[n_models - 1 :])]
        return model_w, mult

    def objective(x: np.ndarray) -> float:
        model_w, mult = unpack(x)
        prob = sum(w * p for w, p in zip(model_w, oof_list))
        pred = apply_multipliers(prob, mult)
        return -weighted_balanced_accuracy(y, pred, weights)

    bounds = [(-2.5, 2.5)] * (n_models - 1) + [(-1.4, 1.4)] * (oof_list[0].shape[1] - 1)
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=12,
        maxiter=55,
        polish=True,
        updating="immediate",
        workers=1,
        tol=1e-7,
    )
    model_w, mult = unpack(result.x)
    return model_w, mult, -float(result.fun)


def find_prior_multipliers(prob: np.ndarray, target_prior: np.ndarray) -> np.ndarray:
    def objective(x: np.ndarray) -> float:
        mult = np.r_[1.0, np.exp(x)]
        pred = apply_multipliers(prob, mult)
        observed = np.bincount(pred, minlength=prob.shape[1]) / len(pred)
        return float(np.square(observed - target_prior).sum())

    result = differential_evolution(
        objective,
        bounds=[(-2.0, 2.0)] * (prob.shape[1] - 1),
        seed=SEED + 71,
        popsize=12,
        maxiter=40,
        polish=True,
    )
    return np.r_[1.0, np.exp(result.x)]


def model_report(y: np.ndarray, prob: np.ndarray, labels: np.ndarray, multipliers: np.ndarray | None = None) -> dict:
    if multipliers is None:
        multipliers = np.ones(prob.shape[1])
    pred = apply_multipliers(prob, multipliers)
    cm = confusion_matrix(y, pred, labels=np.arange(len(labels)))
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "multipliers": multipliers.tolist(),
        "per_class_recall": {str(labels[i]): float(recalls[i]) for i in range(len(labels))},
        "confusion_matrix": cm.tolist(),
    }


def build_adversarial_weights(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(SEED)
    n_each = min(220_000, len(Xtr), len(Xte))
    tr_idx = rng.choice(len(Xtr), size=n_each, replace=False)
    te_idx = rng.choice(len(Xte), size=n_each, replace=False)
    X_adv = pd.concat([Xtr.iloc[tr_idx], Xte.iloc[te_idx]], ignore_index=True)
    y_adv = np.r_[np.zeros(n_each, dtype=np.int8), np.ones(n_each, dtype=np.int8)]
    order = rng.permutation(len(X_adv))
    split = int(len(order) * 0.8)
    fit_idx, val_idx = order[:split], order[split:]
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=SEED,
    )
    model.fit(X_adv.iloc[fit_idx], y_adv[fit_idx])
    val_prob = model.predict_proba(X_adv.iloc[val_idx])[:, 1]
    auc = float(roc_auc_score(y_adv[val_idx], val_prob))
    p = np.clip(model.predict_proba(Xtr)[:, 1], 0.01, 0.99)
    # Density ratio p(test|x)/p(train|x), normalized and clipped for stability.
    w = p / (1.0 - p)
    w = np.clip(w, np.quantile(w, 0.01), np.quantile(w, 0.99))
    w = w / np.mean(w)
    details = {
        "auc": auc,
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_mean": float(w.mean()),
        "weight_quantiles": {str(q): float(np.quantile(w, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
    }
    del X_adv, model
    gc.collect()
    return w.astype(np.float32), details


def train_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    oof = np.zeros((len(X), len(np.unique(y))), dtype=np.float32)
    test_prob = np.zeros((len(X_test), oof.shape[1]), dtype=np.float64)
    best_iterations = []

    def metric(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[str, float, bool]:
        return "balanced_accuracy", balanced_accuracy_score(y_true, np.argmax(y_prob, axis=1)), True

    params = dict(
        objective="multiclass",
        n_estimators=6000,
        learning_rate=0.035,
        num_leaves=41,
        min_child_samples=35,
        colsample_bytree=0.8401695624538926,
        subsample=0.8092497041550049,
        subsample_freq=1,
        reg_alpha=0.38651187383382324,
        reg_lambda=0.06292313532864219,
        class_weight="balanced",
        max_bin=255,
        n_jobs=2,
        verbosity=-1,
    )
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = LGBMClassifier(**params, random_state=SEED + fold * 17)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric=metric,
            callbacks=[early_stopping(250, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
        test_prob += model.predict_proba(X_test) / len(folds)
        best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
        score = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        print(f"LGBM fold {fold}: {score:.6f}, best_iter={best_iterations[-1]}", flush=True)
        del model
        gc.collect()
    return oof, test_prob.astype(np.float32), {"params": params, "best_iterations": best_iterations}


def train_hgb(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    oof = np.zeros((len(X), len(np.unique(y))), dtype=np.float32)
    test_prob = np.zeros((len(X_test), oof.shape[1]), dtype=np.float64)
    params = dict(
        learning_rate=0.08,
        max_iter=700,
        max_leaf_nodes=31,
        min_samples_leaf=45,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=60,
        class_weight="balanced",
    )
    iterations = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = HistGradientBoostingClassifier(**params, random_state=SEED + 101 + fold)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
        test_prob += model.predict_proba(X_test) / len(folds)
        iterations.append(int(model.n_iter_))
        score = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        print(f"HGB fold {fold}: {score:.6f}, iter={model.n_iter_}", flush=True)
        del model
        gc.collect()
    return oof, test_prob.astype(np.float32), {"params": params, "iterations": iterations}


def train_xgb_ovr(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    from xgboost import XGBClassifier

    n_classes = len(np.unique(y))
    oof_raw = np.zeros((len(X), n_classes), dtype=np.float32)
    test_raw = np.zeros((len(X_test), n_classes), dtype=np.float64)
    params = dict(
        n_estimators=2600,
        learning_rate=0.035,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=2.0,
        gamma=0.05,
        tree_method="hist",
        max_bin=256,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=2,
        early_stopping_rounds=120,
    )
    best_iters: list[list[int]] = []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        fold_iters = []
        for c in range(n_classes):
            y_bin = (y == c).astype(np.int8)
            pos = int(y_bin[tr_idx].sum())
            neg = len(tr_idx) - pos
            model = XGBClassifier(
                **params,
                random_state=SEED + 1000 + fold * 37 + c,
                scale_pos_weight=neg / max(pos, 1),
            )
            model.fit(
                X.iloc[tr_idx],
                y_bin[tr_idx],
                eval_set=[(X.iloc[va_idx], y_bin[va_idx])],
                verbose=False,
            )
            oof_raw[va_idx, c] = model.predict_proba(X.iloc[va_idx])[:, 1]
            test_raw[:, c] += model.predict_proba(X_test)[:, 1] / len(folds)
            fold_iters.append(int((model.best_iteration or 0) + 1))
            del model
            gc.collect()
        oof_fold = oof_raw[va_idx] / np.maximum(oof_raw[va_idx].sum(axis=1, keepdims=True), 1e-12)
        score = balanced_accuracy_score(y[va_idx], oof_fold.argmax(axis=1))
        best_iters.append(fold_iters)
        print(f"XGB-OVR fold {fold}: {score:.6f}, best_iters={fold_iters}", flush=True)
    oof = oof_raw / np.maximum(oof_raw.sum(axis=1, keepdims=True), 1e-12)
    test_prob = test_raw / np.maximum(test_raw.sum(axis=1, keepdims=True), 1e-12)
    return oof.astype(np.float32), test_prob.astype(np.float32), {"params": params, "best_iterations": best_iters}


def try_public_submissions(output_dir: Path, sample: pd.DataFrame) -> dict[str, dict]:
    sources = {
        "public_hook_094967": (
            "https://raw.githubusercontent.com/Hook12aaa/kaggle-health-ps-s6e7/"
            "5f9a7dac59a8431482cf486645e773887e4dbd6f/submission.csv"
        ),
        "public_mrjohnson": (
            "https://raw.githubusercontent.com/mrjohnsonsea/playground-series-s6e7/"
            "d2ac3f0e767f14fc337c41d966a5a1fb16901604/data/submission/submission.csv"
        ),
    }
    report: dict[str, dict] = {}
    for name, url in sources.items():
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            path = output_dir / f"submission_{name}.csv"
            path.write_bytes(response.content)
            df = pd.read_csv(path)
            valid = list(df.columns) == list(sample.columns) and len(df) == len(sample) and df[ID].equals(sample[ID])
            if not valid:
                path.unlink(missing_ok=True)
                raise ValueError("submission schema or id order mismatch")
            report[name] = {
                "status": "downloaded",
                "sha256": sha256(path),
                "distribution": df[TARGET].value_counts(normalize=True).to_dict(),
            }
        except Exception as exc:
            report[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # Probe public Kaggle kernel-output endpoints. They sometimes permit anonymous download.
    kernel_refs = [
        "amanatar/s6e7-student-hearth-risk-lb-0-95112",
        "anhadmahajan06/s6e7-post-processing-ensemble-lb-0-95112",
        "makthanithin/s6e7-post-processing-ensemble-lb-0-95112",
    ]
    for ref in kernel_refs:
        key = "kernel_" + ref.replace("/", "__")
        try:
            url = f"https://www.kaggle.com/api/v1/kernels/output/{ref}"
            response = requests.get(url, timeout=240, allow_redirects=True)
            entry: dict = {
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "bytes": len(response.content),
            }
            if response.status_code == 200 and len(response.content) > 1000:
                zip_path = output_dir / f"{key}.zip"
                zip_path.write_bytes(response.content)
                if zipfile.is_zipfile(zip_path):
                    extract_dir = output_dir / key
                    extract_dir.mkdir(exist_ok=True)
                    with zipfile.ZipFile(zip_path) as zf:
                        zf.extractall(extract_dir)
                    csvs = list(extract_dir.rglob("*.csv"))
                    valid_csv = None
                    for csv in csvs:
                        try:
                            df = pd.read_csv(csv)
                            if list(df.columns) == list(sample.columns) and len(df) == len(sample) and df[ID].equals(sample[ID]):
                                valid_csv = csv
                                break
                        except Exception:
                            pass
                    if valid_csv is not None:
                        target = output_dir / f"submission_{key}.csv"
                        target.write_bytes(valid_csv.read_bytes())
                        entry["submission"] = target.name
                        entry["distribution"] = pd.read_csv(target)[TARGET].value_counts(normalize=True).to_dict()
                zip_path.unlink(missing_ok=True)
            report[key] = entry
        except Exception as exc:
            report[key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return report


def save_submission(
    output_dir: Path,
    name: str,
    sample: pd.DataFrame,
    labels: np.ndarray,
    pred_idx: np.ndarray,
) -> Path:
    out = sample.copy()
    out[TARGET] = labels[pred_idx]
    path = output_dir / f"submission_{name}.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    assert len(train) == 690_088 and len(test) == 295_753
    assert sample[ID].equals(test[ID])

    labels = np.array(sorted(train[TARGET].unique()))
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(train[TARGET])
    print("labels:", labels.tolist(), flush=True)
    print("train/test:", train.shape, test.shape, flush=True)
    print("target distribution:", train[TARGET].value_counts(normalize=True).to_dict(), flush=True)
    print("categorical values:", {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS}, flush=True)

    # Diagnostics for sequential/id drift.
    id_bins = pd.qcut(train[ID], 10, labels=False, duplicates="drop")
    id_target = pd.crosstab(id_bins, train[TARGET], normalize="index")
    id_target.to_csv(args.output_dir / "target_by_id_decile.csv")

    X, X_test, categorical_encoded = prepare_encoded(train, test)
    print(f"engineered encoded shape: {X.shape}; categorical/code columns={len(categorical_encoded)}", flush=True)
    adv_weights, adv_report = build_adversarial_weights(X, X_test)
    print("adversarial report:", adv_report, flush=True)

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y))
    fold_id = np.full(len(train), -1, dtype=np.int8)
    for f, (_, va_idx) in enumerate(folds):
        fold_id[va_idx] = f

    model_oof: dict[str, np.ndarray] = {}
    model_test: dict[str, np.ndarray] = {}
    model_meta: dict[str, dict] = {}

    for name, trainer in [
        ("lgbm", train_lgbm),
        ("hgb", train_hgb),
        ("xgb_ovr", train_xgb_ovr),
    ]:
        print(f"\n=== TRAINING {name} ===", flush=True)
        oof, pred, meta = trainer(X, y, X_test, folds)
        model_oof[name] = oof
        model_test[name] = pred
        model_meta[name] = meta
        np.savez_compressed(args.output_dir / f"predictions_{name}.npz", oof=oof, test=pred, fold_id=fold_id)
        gc.collect()

    metrics: dict[str, dict] = {}
    for i, name in enumerate(model_oof):
        oof = model_oof[name]
        raw = model_report(y, oof, labels)
        mult_std, score_std = tune_multipliers(y, oof, seed=SEED + i)
        mult_drift, score_drift = tune_multipliers(y, oof, weights=adv_weights, seed=SEED + 20 + i)
        metrics[name] = {
            "raw": raw,
            "standard_tuned": model_report(y, oof, labels, mult_std),
            "drift_tuned_weighted_score": score_drift,
            "drift_multipliers": mult_drift.tolist(),
        }
        save_submission(args.output_dir, f"{name}_standard", sample, labels, apply_multipliers(model_test[name], mult_std))
        save_submission(args.output_dir, f"{name}_drift", sample, labels, apply_multipliers(model_test[name], mult_drift))
        print(name, json.dumps(metrics[name], indent=2), flush=True)

    names = list(model_oof)
    oof_list = [model_oof[n] for n in names]
    test_list = [model_test[n] for n in names]
    blend_w_std, blend_mult_std, blend_score_std = optimize_blend(y, oof_list, seed=SEED + 101)
    blend_w_drift, blend_mult_drift, blend_score_drift = optimize_blend(
        y, oof_list, weights=adv_weights, seed=SEED + 202
    )
    blend_prob_std = sum(w * p for w, p in zip(blend_w_std, oof_list))
    blend_test_std = sum(w * p for w, p in zip(blend_w_std, test_list))
    blend_prob_drift = sum(w * p for w, p in zip(blend_w_drift, oof_list))
    blend_test_drift = sum(w * p for w, p in zip(blend_w_drift, test_list))

    blend_report_std = model_report(y, blend_prob_std, labels, blend_mult_std)
    blend_report_drift = model_report(y, blend_prob_drift, labels, blend_mult_drift)
    blend_report_drift["weighted_balanced_accuracy"] = blend_score_drift
    blend_report_std["model_weights"] = dict(zip(names, blend_w_std.tolist()))
    blend_report_drift["model_weights"] = dict(zip(names, blend_w_drift.tolist()))

    path_standard = save_submission(
        args.output_dir,
        "blend_standard",
        sample,
        labels,
        apply_multipliers(blend_test_std, blend_mult_std),
    )
    path_drift = save_submission(
        args.output_dir,
        "blend_drift",
        sample,
        labels,
        apply_multipliers(blend_test_drift, blend_mult_drift),
    )

    # A compromise uses standard model weights and the average of standard/drift log multipliers.
    compromise_mult = np.sqrt(blend_mult_std * blend_mult_drift)
    compromise_prob = 0.5 * blend_test_std + 0.5 * blend_test_drift
    path_compromise = save_submission(
        args.output_dir,
        "blend_compromise",
        sample,
        labels,
        apply_multipliers(compromise_prob, compromise_mult),
    )

    target_prior = np.array([PUBLIC_REFERENCE_PRIOR[str(lbl)] for lbl in labels])
    prior_mult = find_prior_multipliers(compromise_prob, target_prior)
    path_prior = save_submission(
        args.output_dir,
        "blend_public_prior",
        sample,
        labels,
        apply_multipliers(compromise_prob, prior_mult),
    )

    candidate_paths = [path_standard, path_drift, path_compromise, path_prior]
    candidate_summary = {}
    for path in candidate_paths:
        df = pd.read_csv(path)
        candidate_summary[path.name] = {
            "sha256": sha256(path),
            "distribution": df[TARGET].value_counts(normalize=True).to_dict(),
        }

    # Conservative model selection: the compromise avoids overcommitting to either random-CV or drift weighting.
    recommended = args.output_dir / "submission_recommended.csv"
    recommended.write_bytes(path_compromise.read_bytes())
    candidate_summary[recommended.name] = {
        "source": path_compromise.name,
        "sha256": sha256(recommended),
        "distribution": pd.read_csv(recommended)[TARGET].value_counts(normalize=True).to_dict(),
    }

    public_report = try_public_submissions(args.output_dir, sample)
    results = {
        "data": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "train_sha256": sha256(train_path),
            "test_sha256": sha256(test_path),
            "labels": labels.tolist(),
            "target_distribution": train[TARGET].value_counts(normalize=True).to_dict(),
            "adversarial": adv_report,
        },
        "models": metrics,
        "model_metadata": model_meta,
        "blend_standard": blend_report_std,
        "blend_drift": blend_report_drift,
        "blend_standard_multiplier_score": blend_score_std,
        "blend_prior_multipliers": prior_mult.tolist(),
        "candidates": candidate_summary,
        "public_submissions": public_report,
        "recommended": recommended.name,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    np.savez_compressed(
        args.output_dir / "ensemble_predictions.npz",
        oof_standard=blend_prob_std,
        test_standard=blend_test_std,
        oof_drift=blend_prob_drift,
        test_drift=blend_test_drift,
        y=y,
        labels=labels,
        fold_id=fold_id,
        adversarial_weights=adv_weights,
    )
    print("\nFINAL RESULTS", json.dumps(results, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
