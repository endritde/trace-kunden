"""Churn-Modelle, Hyperparameter-Suche auf den Trainingsdaten und Evaluation auf dem Test-Stichtag."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from trace_kunden.konfig import SEED


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def modelle() -> dict[str, tuple[ClassifierMixin | Pipeline, dict]]:
    """Die drei Modelle mit kleinem Suchraum. Klassenungleichgewicht über Klassengewichte.

    ``n_jobs`` bleibt klein, damit der Arbeitsspeicher (8 GB) nicht knapp wird.
    """
    lr = make_pipeline(FunctionTransformer(_signed_log1p), StandardScaler(),
                       LogisticRegression(class_weight="balanced", max_iter=2000))
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=SEED,
                                n_jobs=2)
    hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, early_stopping=False)
    return {
        "Logistic Regression": (lr, {"logisticregression__C": [0.01, 0.1, 1.0]}),
        "Random Forest": (rf, {"max_depth": [6, 10, None], "min_samples_leaf": [5, 20]}),
        "HistGradientBoosting": (hgb, {"learning_rate": [0.03, 0.1], "max_depth": [3, None],
                                       "max_iter": [150]}),
    }


def trainieren(x: pd.DataFrame, y: pd.Series) -> dict[str, GridSearchCV]:
    """Wählt je Modell die Hyperparameter per 5-facher, geschichteter Kreuzvalidierung (ROC-AUC).

    Die Kreuzvalidierung läuft nur auf dem Trainings-Stichtag; der Test-Stichtag bleibt unberührt.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ergebnis = {}
    for name, (modell, raster) in modelle().items():
        suche = GridSearchCV(modell, raster, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        ergebnis[name] = suche.fit(x, y)
    return ergebnis


@dataclass
class Bewertung:
    """Kennzahlen eines Modells auf dem Test-Stichtag."""

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    precision_top20: float
    konfusion: np.ndarray


def bewerten(y: pd.Series, score: np.ndarray, schwelle: float = 0.5) -> Bewertung:
    """ROC-AUC, PR-AUC, Precision/Recall/F1 an der Schwelle und Precision in den 20 % riskantesten Kunden."""
    vorhersage = (score >= schwelle).astype(int)
    top = np.argsort(-score)[: round(0.2 * len(score))]
    return Bewertung(
        roc_auc=roc_auc_score(y, score),
        pr_auc=average_precision_score(y, score),
        precision=precision_score(y, vorhersage, zero_division=0),
        recall=recall_score(y, vorhersage, zero_division=0),
        f1=f1_score(y, vorhersage, zero_division=0),
        precision_top20=float(np.asarray(y)[top].mean()),
        konfusion=confusion_matrix(y, vorhersage, labels=[0, 1]),
    )


def faustregel_score(x: pd.DataFrame) -> np.ndarray:
    """Baseline ohne Machine Learning: Je länger der letzte Kauf her ist, desto höher das Risiko.

    Auf 0–1 skaliert (Recency / 365), damit eine Schwelle von 0,5 „über ca. 6 Monate kein Kauf“ bedeutet.
    """
    return np.clip(x["recency_tage"].to_numpy() / 365, 0, 1)


def wichtigkeit(modell: ClassifierMixin | Pipeline, x: pd.DataFrame, y: pd.Series,
                wiederholungen: int = 10) -> pd.DataFrame:
    """Permutation Importance auf dem Test-Stichtag: Wie stark sinkt die ROC-AUC, wenn ein Feature
    zufällig vertauscht wird? Modellunabhängig und auf ungesehenen Daten gemessen."""
    r = permutation_importance(modell, x, y, scoring="roc_auc", n_repeats=wiederholungen,
                               random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"mittel": r.importances_mean, "std": r.importances_std}, index=x.columns)
            .sort_values("mittel", ascending=False))
