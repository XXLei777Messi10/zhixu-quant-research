from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant.models.splits import TimeSplit
from quant.qlib_workflow.features import FEATURE_COLUMNS


@dataclass
class TrainedModels:
    ridge: Any
    logistic: Any
    lgb_regression: Any
    lgb_classification: Any
    platt: Any
    feature_columns: list[str]


@dataclass
class ModelManifest:
    version: str
    created_at: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    data_cutoff: str
    feature_count: int
    parameters: dict[str, Any]
    package_versions: dict[str, str]
    metrics: dict[str, float]
    git_commit: str
    data_quality_summary: dict[str, Any]
    content_hash: str


class IdentityCalibrator:
    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities).reshape(-1)
        return np.column_stack([1.0 - p, p])


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=42, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.model.fit(logits, labels)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.model.predict_proba(logits)


def _label_columns(model_config: dict[str, Any]) -> tuple[str, str]:
    labels = model_config.get("label_columns") or {}
    return (
        str(labels.get("regression", "label_regression")),
        str(labels.get("classification", "label_classification")),
    )


def _training_rows(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regression_label: str = "label_regression",
) -> pd.DataFrame:
    mask = frame["trade_date"].between(start, end)
    return frame.loc[mask & frame[regression_label].notna()].copy()


def effective_training_start(
    split: TimeSplit,
    model_config: dict[str, Any],
) -> pd.Timestamp:
    years = model_config.get("training_window_years")
    if years is None:
        return pd.Timestamp(split.train_start)
    years = int(years)
    if years < 3:
        raise ValueError("Training window must be at least three years")
    floor = pd.Timestamp(split.train_end) - pd.DateOffset(years=years)
    return max(pd.Timestamp(split.train_start), floor)


def training_sample_weights(
    train: pd.DataFrame,
    model_config: dict[str, Any],
) -> pd.Series | None:
    config = model_config.get("training_sample_weight")
    if config is None:
        return None
    if str(config.get("method")) != "exponential_time_decay":
        raise ValueError("Unsupported training sample-weight method")
    half_life_years = float(config.get("half_life_years", 0))
    if not np.isfinite(half_life_years) or half_life_years <= 0:
        raise ValueError("Training sample-weight half-life must be positive")
    if str(config.get("normalization", "mean_one")) != "mean_one":
        raise ValueError("Training sample weights must use mean_one normalization")
    dates = pd.to_datetime(train["trade_date"]).dt.normalize()
    age_years = (dates.max() - dates).dt.days.astype(float) / 365.2425
    weights = np.power(0.5, age_years / half_life_years)
    weights = weights / weights.mean()
    return pd.Series(weights.astype("float32"), index=train.index, name="sample_weight")


def lightgbm_regression_parameters(
    base_params: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    objective_config = model_config.get("lightgbm_regression_objective") or {
        "name": "regression"
    }
    objective = str(objective_config.get("name", "regression"))
    if objective not in {"regression", "huber"}:
        raise ValueError(f"Unsupported LightGBM regression objective: {objective}")
    params = {**base_params, "objective": objective, "metric": "l2"}
    if objective == "huber":
        alpha = float(objective_config.get("alpha", 0))
        if not 0 < alpha < 1:
            raise ValueError("Huber alpha must be between zero and one")
        params["alpha"] = alpha
    return params


def fit_models(
    frame: pd.DataFrame,
    split: TimeSplit,
    model_config: dict[str, Any],
) -> tuple[TrainedModels, dict[str, float], dict[str, float]]:
    regression_label, classification_label = _label_columns(model_config)
    missing_labels = {
        regression_label,
        classification_label,
    }.difference(frame.columns)
    if missing_labels:
        raise ValueError(f"Training frame missing label columns: {sorted(missing_labels)}")
    train_start = effective_training_start(split, model_config)
    train = _training_rows(frame, train_start, split.train_end, regression_label)
    valid = _training_rows(frame, split.valid_start, split.valid_end, regression_label)
    if train.empty or valid.empty:
        raise ValueError("Train and validation windows must both contain mature labels")
    features = [column for column in FEATURE_COLUMNS if column in frame.columns]
    x_train = train[features].astype("float32")
    x_valid = valid[features].astype("float32")
    y_train_reg = train[regression_label].astype(float)
    y_valid_reg = valid[regression_label].astype(float)
    y_train_cls = train[classification_label].astype(int)
    y_valid_cls = valid[classification_label].astype(int)
    sample_weights = training_sample_weights(train, model_config)
    linear_fit_params = (
        {"model__sample_weight": sample_weights.to_numpy()}
        if sample_weights is not None
        else {}
    )

    ridge = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    ).fit(x_train, y_train_reg, **linear_fit_params)
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, random_state=42, max_iter=1000)),
        ]
    ).fit(x_train, y_train_cls, **linear_fit_params)

    params = dict(model_config["lightgbm"])
    num_boost_round = int(params.pop("num_boost_round"))
    early_stopping = int(params.pop("early_stopping_rounds"))
    params.update(
        {
            "verbosity": -1,
            "seed": int(model_config["seed"]),
            "feature_fraction_seed": int(model_config["seed"]),
            "bagging_seed": int(model_config["seed"]),
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    reg_params = lightgbm_regression_parameters(params, model_config)
    cls_params = {**params, "objective": "binary", "metric": ["binary_logloss", "auc"]}
    lgb_weights = sample_weights.to_numpy() if sample_weights is not None else None
    train_reg = lgb.Dataset(
        x_train,
        label=y_train_reg,
        weight=lgb_weights,
        feature_name=features,
    )
    valid_reg = lgb.Dataset(x_valid, label=y_valid_reg, feature_name=features, reference=train_reg)
    regression = lgb.train(
        reg_params,
        train_reg,
        num_boost_round=num_boost_round,
        valid_sets=[valid_reg],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    train_cls = lgb.Dataset(
        x_train,
        label=y_train_cls,
        weight=lgb_weights,
        feature_name=features,
    )
    valid_cls = lgb.Dataset(x_valid, label=y_valid_cls, feature_name=features, reference=train_cls)
    classification = lgb.train(
        cls_params,
        train_cls,
        num_boost_round=num_boost_round,
        valid_sets=[valid_cls],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    raw_probability = classification.predict(x_valid)
    if y_valid_cls.nunique() >= 2:
        platt: Any = PlattCalibrator().fit(raw_probability, y_valid_cls.to_numpy())
    else:
        platt = IdentityCalibrator()
    calibrated = platt.predict_proba(raw_probability)[:, 1]
    reg_prediction = regression.predict(x_valid)
    metrics = {
        "label_horizon": float(model_config.get("label_horizon", 5)),
        "train_rows": float(len(train)),
        "training_window_years": float(
            model_config.get("training_window_years") or 0
        ),
        "sample_weight_half_life_years": float(
            (model_config.get("training_sample_weight") or {}).get(
                "half_life_years", 0
            )
        ),
        "sample_weight_min": float(
            sample_weights.min() if sample_weights is not None else 1.0
        ),
        "sample_weight_max": float(
            sample_weights.max() if sample_weights is not None else 1.0
        ),
        "sample_weight_effective_rows": float(
            (
                sample_weights.sum() ** 2
                / sample_weights.pow(2).sum()
            )
            if sample_weights is not None
            else len(train)
        ),
        "valid_regression_correlation": float(
            pd.Series(reg_prediction).corr(pd.Series(y_valid_reg.to_numpy()))
        ),
        "valid_classification_accuracy": float(
            ((calibrated >= 0.5).astype(int) == y_valid_cls.to_numpy()).mean()
        ),
        "valid_brier": float(np.mean((calibrated - y_valid_cls.to_numpy()) ** 2)),
    }
    importance = dict(zip(features, regression.feature_importance(importance_type="gain"), strict=True))
    models = TrainedModels(ridge, logistic, regression, classification, platt, features)
    return models, metrics, {key: float(value) for key, value in importance.items()}


def predict_models(models: TrainedModels, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[["symbol", "trade_date"]].copy()
    output["symbol"] = output["symbol"].astype("string")
    x = frame[models.feature_columns].astype("float32")
    output["ridge_prediction"] = models.ridge.predict(x)
    output["logistic_probability"] = models.logistic.predict_proba(x)[:, 1]
    output["predicted_excess_return"] = models.lgb_regression.predict(x)
    raw_probability = models.lgb_classification.predict(x)
    output["outperform_probability"] = models.platt.predict_proba(raw_probability)[:, 1]
    output["regression_rank"] = output.groupby("trade_date")["predicted_excess_return"].rank(pct=True)
    output["classification_rank"] = output.groupby("trade_date")["outperform_probability"].rank(pct=True)
    output["model_score"] = 0.5 * output["regression_rank"] + 0.5 * output["classification_rank"]
    return output


def save_model_bundle(
    models: TrainedModels,
    split: TimeSplit,
    model_config: dict[str, Any],
    metrics: dict[str, float],
    importance: dict[str, float],
    root: Path,
    data_quality_summary: dict[str, Any] | None = None,
) -> Path:
    basis = json.dumps(
        {
            "split": {key: str(value) for key, value in asdict(split).items()},
            "config": model_config,
            "features": models.feature_columns,
        },
        sort_keys=True,
        default=str,
    )
    version = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(basis.encode()).hexdigest()[:10]
    )
    bundle = root / version
    bundle.mkdir(parents=True, exist_ok=False)
    joblib.dump(models, bundle / "models.joblib", compress=3)
    packages = {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    project_root = root.parents[1]
    git_commit = _git_commit(project_root)
    content_hash = hashlib.sha256(
        json.dumps(
            {
                "basis": basis,
                "metrics": metrics,
                "importance": importance,
                "packages": packages,
                "git_commit": git_commit,
                "data_quality_summary": data_quality_summary or {},
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    manifest = ModelManifest(
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        train_start=str(split.train_start.date()),
        train_end=str(split.train_end.date()),
        valid_start=str(split.valid_start.date()),
        valid_end=str(split.valid_end.date()),
        data_cutoff=str(split.valid_end.date()),
        feature_count=len(models.feature_columns),
        parameters=model_config,
        package_versions=packages,
        metrics=metrics,
        git_commit=git_commit,
        data_quality_summary=data_quality_summary or {},
        content_hash=content_hash,
    )
    (bundle / "manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (bundle / "feature_importance.json").write_text(
        json.dumps(importance, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    pointer = root / "current.json"
    temp = pointer.with_name(f".current.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"version": version, "path": str(bundle)}, indent=2), encoding="utf-8")
    os.replace(temp, pointer)
    return bundle


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={project_root.as_posix()}",
                "rev-parse",
                "--show-toplevel",
                "HEAD",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        result = None
    if result is not None:
        lines = result.stdout.splitlines()
        if (
            len(lines) == 2
            and Path(lines[0]).resolve() == project_root.resolve()
            and len(lines[1].strip()) == 40
        ):
            return lines[1].strip()
    build_commit = project_root / "BUILD_COMMIT"
    if build_commit.exists():
        value = build_commit.read_text(encoding="utf-8").strip()
        if len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower()):
            return value
    return "UNVERSIONED"


def load_current_models(root: Path) -> tuple[TrainedModels, dict[str, Any]]:
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    bundle = Path(pointer["path"])
    models = joblib.load(bundle / "models.joblib")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return models, manifest
