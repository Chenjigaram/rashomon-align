import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rashomon_align.tabular import cross_validated_comparison

warnings.filterwarnings("ignore")

DATASETS = [
    "iris", "wine", "breast-w", "diabetes", "sonar", "ionosphere", "glass", "vehicle",
    "segment", "ecoli", "balance-scale", "credit-g", "credit-a", "vowel", "haberman",
    "heart-statlog", "liver-disorders", "tic-tac-toe", "car", "blood-transfusion-service-center",
    "banknote-authentication", "climate-model-simulation-crashes", "cmc", "wdbc",
    "planning-relax", "qsar-biodeg", "steel-plates-fault", "phoneme", "waveform-5000", "kc1",
]


def unpruned():
    return DecisionTreeClassifier(min_samples_split=2, random_state=42)


def pruned():
    return DecisionTreeClassifier(min_samples_split=10, ccp_alpha=0.01, random_state=42)


def prepare(frame, target):
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    transformer = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    X = transformer.fit_transform(frame)
    y = pd.Series(target).astype("category").cat.codes.to_numpy()
    return np.asarray(X, dtype=float), y


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the Rashomon Alignment tabular protocol")
    parser.add_argument("--out", type=Path, default=Path("results/reproduction.json"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = []
    for name in DATASETS[: args.limit]:
        try:
            data = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
            X, y = prepare(data.data, data.target)
            if len(np.unique(y)) < 2 or X.shape[0] < 40:
                print(f"skip {name}: unusable")
                continue
            folds = cross_validated_comparison(unpruned, pruned, X, y, folds=5, seed=42, count=args.samples)
        except Exception as error:
            print(f"skip {name}: {type(error).__name__}")
            continue
        row = {
            "dataset": name,
            "instances": int(X.shape[0]),
            "features": int(X.shape[1]),
            "accuracy_difference": float(np.mean([f.accuracy_difference for f in folds])),
            "dra": float(np.mean([f.dra for f in folds])),
            "gra": float(np.mean([f.gra for f in folds])),
        }
        rows.append(row)
        print(f"{name:<42} dRA={row['dra']:.3f}  gRA={row['gra']:.3f}  dAcc={row['accuracy_difference']:.3f}")

    if len(rows) < 3:
        raise SystemExit("too few datasets succeeded to compute correlations")

    dra = np.array([r["dra"] for r in rows])
    gra = np.array([r["gra"] for r in rows])
    dacc = np.array([r["accuracy_difference"] for r in rows])
    summary = {
        "datasets": len(rows),
        "mean_dra": float(dra.mean()),
        "mean_gra": float(gra.mean()),
        "pearson_gra_dra": float(pearsonr(gra, dra)[0]),
        "pearson_gra_accuracy_difference": float(pearsonr(gra, dacc)[0]),
        "paper_pearson_gra_dra": 0.745,
        "paper_pearson_gra_accuracy_difference": 0.514,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print()
    print(f"datasets            {summary['datasets']}")
    print(f"mean dRA            {summary['mean_dra']:.3f}")
    print(f"mean gRA            {summary['mean_gra']:.3f}   (paper: dRA concentrates above gRA)")
    print(f"r(gRA, dRA)         {summary['pearson_gra_dra']:.3f}   (paper: 0.745)")
    print(f"r(gRA, delta acc)   {summary['pearson_gra_accuracy_difference']:.3f}   (paper: 0.514)")


if __name__ == "__main__":
    main()
