from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from loyalty_engine.models.artifacts import ModelBundle


def _make_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


TARGET_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "customer_health",
        "churn_risk",
        "customer_health_score",
        "churn_risk_score",
        "loyalty_score",
    }
)


def _is_leakage_feature(column_name: str) -> bool:
    """Return True for features that leak target information from health/churn labels."""
    normalized = column_name.strip().lower()
    if normalized in TARGET_LEAKAGE_COLUMNS:
        return True
    # Catch any additional direct derivatives of the supervised targets.
    return normalized.startswith("customer_health_") or normalized.startswith("churn_risk_")


def infer_feature_schema(dataset: pd.DataFrame, target_columns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Infer numeric vs categorical feature columns from the training table."""
    excluded = {"Customer_ID", *target_columns}
    feature_columns = [
        col for col in dataset.columns if col not in excluded and not _is_leakage_feature(col)
    ]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in feature_columns:
        series = dataset[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def _build_classifier(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _make_preprocessor(numeric_columns, categorical_columns)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _split_train_test_with_stratification_guard(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data with stratification, keeping singleton classes in the training fold.

    `train_test_split(..., stratify=y)` is used when every class has at least two
    members. If a class appears only once, stratification is mathematically
    impossible, so that singleton row is retained in training and the remaining
    rows are split with stratification.
    """
    class_counts = y.value_counts(dropna=False)
    singleton_labels = class_counts[class_counts < 2].index

    if len(singleton_labels) == 0:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    singleton_mask = y.isin(singleton_labels)
    X_singletons = X.loc[singleton_mask]
    y_singletons = y.loc[singleton_mask]
    X_regular = X.loc[~singleton_mask]
    y_regular = y.loc[~singleton_mask]

    if y_regular.empty:
        raise ValueError("Not enough non-singleton samples to create a holdout split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_regular,
        y_regular,
        test_size=test_size,
        random_state=random_state,
        stratify=y_regular,
    )

    X_train = pd.concat([X_train, X_singletons], axis=0)
    y_train = pd.concat([y_train, y_singletons], axis=0)
    return X_train, X_test, y_train, y_test


def _evaluate_classifier(
    classifier: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float], str]:
    X_train, X_test, y_train, y_test = _split_train_test_with_stratification_guard(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    model_labels = list(classifier.named_steps["model"].classes_)
    metrics = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model_labels).tolist(),
    }
    report = classification_report(y_test, y_pred)
    return classifier, metrics, report


def train_customer_models(
    dataset: pd.DataFrame,
    target_columns: tuple[str, ...],
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> tuple[ModelBundle, dict[str, str], dict[str, dict[str, float]]]:
    if numeric_columns is None or categorical_columns is None:
        numeric_columns, categorical_columns = infer_feature_schema(dataset, target_columns)

    excluded_columns = set(target_columns) | {"Customer_ID"}
    feature_columns = [
        col for col in dataset.columns if col not in excluded_columns and not _is_leakage_feature(col)
    ]
    X = dataset[feature_columns]

    churn_model = _build_classifier(numeric_columns, categorical_columns)
    health_model = _build_classifier(numeric_columns, categorical_columns)

    churn_model, churn_metrics, churn_report = _evaluate_classifier(
        churn_model,
        X,
        dataset[target_columns[0]],
    )
    health_model, health_metrics, health_report = _evaluate_classifier(
        health_model,
        X,
        dataset[target_columns[1]],
    )

    bundle = ModelBundle(
        churn_model=churn_model,
        health_model=health_model,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        metadata={"target_columns": target_columns},
    )

    predictions = {
        "churn_report": churn_report,
        "health_report": health_report,
    }
    evaluation = {
        "churn_model": churn_metrics,
        "health_model": health_metrics,
    }
    return bundle, predictions, evaluation
