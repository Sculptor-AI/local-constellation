from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ID = "id"
TARGET = "health_condition"
CLASSES = np.array(["at-risk", "fit", "unhealthy"])
NUMERIC = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CATEGORICAL = [
    "diet_type", "stress_level", "sleep_quality", "physical_activity_level",
    "smoking_alcohol", "gender",
]
SEED = 20260713


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_hgb(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = train[NUMERIC].copy()
    Xt = test[NUMERIC].copy()
    for c in CATEGORICAL:
        categories = pd.Categorical(pd.concat([train[c], test[c]], ignore_index=True)).categories
        a = pd.Categorical(train[c], categories=categories).codes.astype("float64")
        b = pd.Categorical(test[c], categories=categories).codes.astype("float64")
        a[a == -1] = np.nan
        b[b == -1] = np.nan
        X[c] = a
        Xt[c] = b
    return X, Xt


def encode_lgb(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = train[NUMERIC].copy()
    Xt = test[NUMERIC].copy()
    for c in CATEGORICAL:
        categories = pd.Categorical(pd.concat([train[c], test[c]], ignore_index=True)).categories
        X[c] = pd.Categorical(train[c], categories=categories)
        Xt[c] = pd.Categorical(test[c], categories=categories)
    return X, Xt


def tune_multipliers(prob: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(x: np.ndarray) -> float:
        mult = np.r_[1.0, np.exp(x)]
        pred = CLASSES[(prob * mult).argmax(axis=1)]
        return -balanced_accuracy_score(y, pred)

    result = differential_evolution(
        objective,
        bounds=[(-1.8, 1.8), (-1.8, 1.8)],
        seed=SEED,
        maxiter=60,
        popsize=15,
        polish=True,
        tol=1e-8,
    )
    mult = np.r_[1.0, np.exp(result.x)]
    return mult, -float(result.fun)


def tune_blend(oofs: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(oofs)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = np.r_[x[: n - 1], 0.0]
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        mult = np.r_[1.0, np.exp(x[n - 1 :])]
        return weights, mult

    def objective(x: np.ndarray) -> float:
        weights, mult = unpack(x)
        prob = sum(w * p for w, p in zip(weights, oofs))
        pred = CLASSES[(prob * mult).argmax(axis=1)]
        return -balanced_accuracy_score(y, pred)

    result = differential_evolution(
        objective,
        bounds=[(-2.0, 2.0)] * (n - 1) + [(-1.8, 1.8), (-1.8, 1.8)],
        seed=SEED + 1,
        maxiter=75,
        popsize=15,
        polish=True,
        tol=1e-8,
    )
    weights, mult = unpack(result.x)
    return weights, mult, -float(result.fun)


def report(prob: np.ndarray, y: np.ndarray, mult: np.ndarray) -> dict:
    pred = CLASSES[(prob * mult).argmax(axis=1)]
    cm = confusion_matrix(y, pred, labels=CLASSES)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "multipliers": mult.tolist(),
        "per_class_recall": {
            str(CLASSES[i]): float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }


def save_submission(path: Path, sample: pd.DataFrame, prob: np.ndarray, mult: np.ndarray) -> dict:
    out = sample.copy()
    out[TARGET] = CLASSES[(prob * mult).argmax(axis=1)]
    out.to_csv(path, index=False)
    return {
        "sha256": sha256(path),
        "distribution": out[TARGET].value_counts(normalize=True).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    assert train.shape == (690088, 15)
    assert test.shape == (295753, 14)
    assert sample[ID].equals(test[ID])
    y = train[TARGET].to_numpy()

    Xh, Xth = encode_hgb(train, test)
    Xl, Xtl = encode_lgb(train, test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(Xh, y))

    member_specs = [
        ("hgb_c1_s42", "hgb", 42),
        ("hgb_c1_s7", "hgb", 7),
        ("hgb_c1_s2026", "hgb", 2026),
        ("lgb_small_s42", "lgb", 42),
        ("lgb_small_s7", "lgb", 7),
    ]
    oofs = {name: np.zeros((len(train), 3), dtype=np.float32) for name, _, _ in member_specs}
    tests = {name: np.zeros((len(test), 3), dtype=np.float64) for name, _, _ in member_specs}
    fold_scores: dict[str, list[float]] = {name: [] for name, _, _ in member_specs}

    for fold, (tr, va) in enumerate(splits):
        sw = compute_sample_weight("balanced", y[tr])
        for name, kind, seed in member_specs:
            t0 = time.time()
            if kind == "hgb":
                model = HistGradientBoostingClassifier(
                    max_iter=1000,
                    learning_rate=0.05,
                    max_leaf_nodes=63,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=seed + fold * 100,
                )
                model.fit(Xh.iloc[tr], y[tr], sample_weight=sw)
                val_prob = model.predict_proba(Xh.iloc[va])
                test_prob = model.predict_proba(Xth)
                iterations = int(model.n_iter_)
            else:
                model = LGBMClassifier(
                    objective="multiclass",
                    class_weight="balanced",
                    num_leaves=31,
                    learning_rate=0.08,
                    n_estimators=240,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    reg_lambda=0.1,
                    n_jobs=2,
                    random_state=seed + fold * 100,
                    verbosity=-1,
                )
                model.fit(Xl.iloc[tr], y[tr])
                val_prob = model.predict_proba(Xl.iloc[va])
                test_prob = model.predict_proba(Xtl)
                iterations = int(model.n_estimators_)
            assert np.array_equal(model.classes_, CLASSES)
            oofs[name][va] = val_prob.astype(np.float32)
            tests[name] += test_prob / len(splits)
            score = balanced_accuracy_score(y[va], CLASSES[val_prob.argmax(axis=1)])
            fold_scores[name].append(float(score))
            print(
                f"fold={fold} member={name} score={score:.6f} iter={iterations} "
                f"seconds={time.time() - t0:.1f}",
                flush=True,
            )
            del model, val_prob, test_prob
            gc.collect()

    results: dict = {
        "members": {},
        "fold_scores": fold_scores,
        "data": {
            "train_sha256": sha256(args.data_dir / "train.csv"),
            "test_sha256": sha256(args.data_dir / "test.csv"),
            "target_distribution": train[TARGET].value_counts(normalize=True).to_dict(),
        },
    }

    for name in oofs:
        mult, tuned = tune_multipliers(oofs[name], y)
        results["members"][name] = {
            "raw_balanced_accuracy": float(balanced_accuracy_score(y, CLASSES[oofs[name].argmax(axis=1)])),
            "tuned": report(oofs[name], y, mult),
        }
        results["members"][name]["submission"] = save_submission(
            args.output_dir / f"submission_{name}.csv", sample, tests[name], mult
        )

    names = list(oofs)
    equal_oof = np.mean([oofs[n] for n in names], axis=0)
    equal_test = np.mean([tests[n] for n in names], axis=0)
    equal_mult, _ = tune_multipliers(equal_oof, y)
    results["equal_blend"] = report(equal_oof, y, equal_mult)
    results["equal_blend"]["submission"] = save_submission(
        args.output_dir / "submission_equal_blend.csv", sample, equal_test, equal_mult
    )

    blend_weights, blend_mult, blend_score = tune_blend([oofs[n] for n in names], y)
    tuned_oof = sum(w * oofs[n] for w, n in zip(blend_weights, names))
    tuned_test = sum(w * tests[n] for w, n in zip(blend_weights, names))
    results["optimized_blend"] = report(tuned_oof, y, blend_mult)
    results["optimized_blend"]["model_weights"] = dict(zip(names, blend_weights.tolist()))
    results["optimized_blend"]["optimizer_score"] = blend_score
    results["optimized_blend"]["submission"] = save_submission(
        args.output_dir / "submission_optimized_blend.csv", sample, tuned_test, blend_mult
    )

    # The proven public recipe is the two HGBC seeds plus one LightGBM member.
    proven_names = ["hgb_c1_s42", "hgb_c1_s7", "lgb_small_s42"]
    proven_oof = np.mean([oofs[n] for n in proven_names], axis=0)
    proven_test = np.mean([tests[n] for n in proven_names], axis=0)
    proven_mult, _ = tune_multipliers(proven_oof, y)
    results["proven_three_member"] = report(proven_oof, y, proven_mult)
    results["proven_three_member"]["members"] = proven_names
    results["proven_three_member"]["submission"] = save_submission(
        args.output_dir / "submission_proven_three_member.csv", sample, proven_test, proven_mult
    )

    # Recommended: breadth-averaged five-member blend unless its OOF is materially worse.
    if results["equal_blend"]["balanced_accuracy"] + 0.0002 >= results["proven_three_member"]["balanced_accuracy"]:
        source = args.output_dir / "submission_equal_blend.csv"
        source_name = "equal_blend"
    else:
        source = args.output_dir / "submission_proven_three_member.csv"
        source_name = "proven_three_member"
    recommended = args.output_dir / "submission_recommended_proven.csv"
    recommended.write_bytes(source.read_bytes())
    results["recommended"] = {
        "source": source_name,
        "file": recommended.name,
        "sha256": sha256(recommended),
        "distribution": pd.read_csv(recommended)[TARGET].value_counts(normalize=True).to_dict(),
    }

    np.savez_compressed(
        args.output_dir / "proven_predictions.npz",
        y=y,
        classes=CLASSES,
        **{f"oof_{n}": oofs[n] for n in names},
        **{f"test_{n}": tests[n].astype(np.float32) for n in names},
    )
    results["elapsed_seconds"] = time.time() - started
    (args.output_dir / "results_proven.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
