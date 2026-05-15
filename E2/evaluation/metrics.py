"""Centralized evaluation metrics for the CL vs SFT comparison.

Every classification runner (run_phase{2,3,4}.py, run_e0_zeroshot.py,
experiments/methods/*) should call into this module instead of
re-implementing accuracy / classification_report / confusion_matrix
inline. Aggregating across the 184-run must-have phase requires that
every run emits the same metric schema.

Public API:
    compute_classification_metrics(y_true, y_pred, label_names) -> dict
    compute_ood_scores(...)                                     -> dict
    majority_class_check(y_pred, label_names)                   -> Optional[str]
    paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=10000)  -> dict

Schema returned by compute_classification_metrics (stable, do not break):
    {
      "accuracy": float,
      "f1_macro": float,
      "f1_weighted": float,
      "f1_micro": float,
      "n_samples": int,
      "n_classes_present_true":   int,
      "n_classes_present_pred":   int,
      "per_class": {label_name: {precision, recall, f1, support}},
      "confusion_matrix":        list[list[int]],   # row=true, col=pred
      "confusion_matrix_labels": list[str],          # row/col order
      "majority_class_warning":  Optional[str],
    }
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Sequence[str],
) -> dict:
    """Compute accuracy, F1 (macro / weighted / micro), per-class P/R/F1,
    and a row-major confusion matrix locked to ``label_names``.

    All averages are computed with ``zero_division=0`` and ``labels=label_names``
    so that absent classes are still represented (avoids the silent-drop
    behavior of sklearn's default macro denominator).

    The majority-class warning threshold is task-adaptive (IM-7): with two
    label names (binary) it tightens to 70% so a degenerate "predict
    success for everything" model on a 50/50 split does fire, even though
    50% is below the default 90%.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
    )

    label_names = list(label_names)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) length mismatch"
        )

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(
        y_true, y_pred, average="macro", labels=label_names, zero_division=0,
    ))
    f1_weighted = float(f1_score(
        y_true, y_pred, average="weighted", labels=label_names, zero_division=0,
    ))
    f1_micro = float(f1_score(
        y_true, y_pred, average="micro", labels=label_names, zero_division=0,
    ))

    cls_report = classification_report(
        y_true, y_pred, labels=label_names, zero_division=0, output_dict=True,
    )
    per_class: dict[str, dict[str, float]] = {}
    for name in label_names:
        rec = cls_report.get(name, {})
        per_class[name] = {
            "precision": float(rec.get("precision", 0.0)),
            "recall": float(rec.get("recall", 0.0)),
            "f1": float(rec.get("f1-score", 0.0)),
            "support": int(rec.get("support", 0)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    # IM-7: tighten the threshold for binary tasks so a degenerate
    # "predict the majority class" detector trips at 70% instead of 90%.
    threshold = 0.70 if len(label_names) == 2 else 0.90

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
        "n_samples": len(y_true),
        "n_classes_present_true": len(set(y_true)),
        "n_classes_present_pred": len(set(y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(label_names),
        "majority_class_warning": majority_class_check(
            y_pred, label_names, threshold=threshold),
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def majority_class_check(
    y_pred: Sequence[str],
    label_names: Optional[Sequence[str]] = None,
    threshold: float = 0.9,
) -> Optional[str]:
    """Return a warning string if a single class dominates predictions.

    When >threshold (default 90%) of predictions land on the same label, the
    model is degenerate (predicting the prior). This is the most common silent
    failure mode in classification training. Call this after every eval and
    write the result into your metrics JSON.
    """
    if not y_pred:
        return None
    counts = Counter(y_pred)
    most_common, most_count = counts.most_common(1)[0]
    frac = most_count / len(y_pred)
    if frac > threshold:
        return (
            f"DEGENERATE: {most_count}/{len(y_pred)} predictions ({frac:.1%}) "
            f"= {most_common!r}. Model may be predicting the majority class."
        )
    return None


# ---------------------------------------------------------------------------
# OOD detection scores
# ---------------------------------------------------------------------------

def compute_ood_scores(
    *,
    in_logits: Optional[np.ndarray] = None,
    out_logits: Optional[np.ndarray] = None,
    in_embeddings: Optional[np.ndarray] = None,
    out_embeddings: Optional[np.ndarray] = None,
    train_centroids: Optional[np.ndarray] = None,
    method: str = "msp",
) -> dict:
    """Compute OOD detection AUROC and FPR@95TPR.

    Convention: in-distribution samples have HIGH scores (= the model is
    confident in the prediction), OOD samples have LOW scores. We then
    compute AUROC where the OOD label is 1 (positive class is "out").

    Methods:
      - "msp":      negative max softmax probability  (needs in_logits/out_logits)
      - "energy":   negative log-sum-exp of logits    (needs in_logits/out_logits)
      - "maxlogit": negative max logit                (needs in_logits/out_logits)
      - "centroid": min cosine distance to a class centroid
                    (needs in_embeddings/out_embeddings/train_centroids,
                     all L2-normalized)
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    method = method.lower()
    if method in {"msp", "energy", "maxlogit"}:
        if in_logits is None or out_logits is None:
            raise ValueError(f"method={method!r} requires in_logits and out_logits")
        if method == "msp":
            in_score = -_softmax(in_logits).max(axis=1)
            out_score = -_softmax(out_logits).max(axis=1)
        elif method == "energy":
            in_score = -_logsumexp(in_logits)
            out_score = -_logsumexp(out_logits)
        else:  # maxlogit
            in_score = -in_logits.max(axis=1)
            out_score = -out_logits.max(axis=1)
    elif method == "centroid":
        if in_embeddings is None or out_embeddings is None or train_centroids is None:
            raise ValueError(
                "method='centroid' requires in_embeddings, out_embeddings, "
                "train_centroids"
            )
        in_score = -(in_embeddings @ train_centroids.T).max(axis=1)
        out_score = -(out_embeddings @ train_centroids.T).max(axis=1)
    else:
        raise ValueError(f"Unknown OOD method: {method!r}")

    y = np.concatenate([np.zeros(len(in_score)), np.ones(len(out_score))])
    s = np.concatenate([in_score, out_score])
    auroc = float(roc_auc_score(y, s))

    fpr, tpr, _ = roc_curve(y, s)
    fpr_at_95tpr = float(fpr[np.searchsorted(tpr, 0.95)]) if (tpr >= 0.95).any() else 1.0

    return {
        "method": method,
        "auroc": auroc,
        "fpr_at_95_tpr": fpr_at_95tpr,
        "n_in": int(len(in_score)),
        "n_out": int(len(out_score)),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logsumexp(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(axis=1)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of (scores_a - scores_b).

    Use when you have N seeds of the same experiment for two methods and want
    a CI on the per-seed difference. Returns mean diff, CI, and a two-sided
    p-value (= 2 * fraction of bootstrap diffs that crossed zero).

    Requires len(scores_a) == len(scores_b); the i-th elements are paired.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a {a.shape} and scores_b {b.shape} must have the same shape"
        )
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[k] = diffs[idx].mean()
    lo = float(np.quantile(boot_means, (1 - ci) / 2))
    hi = float(np.quantile(boot_means, 1 - (1 - ci) / 2))

    # Two-sided p-value via the bootstrap distribution.
    if diffs.mean() >= 0:
        p_one = float((boot_means <= 0).mean())
    else:
        p_one = float((boot_means >= 0).mean())
    p_two = min(1.0, 2.0 * p_one)

    return {
        "n_pairs": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)) if n > 1 else 0.0,
        "ci_low": lo,
        "ci_high": hi,
        "ci_level": float(ci),
        "p_value_two_sided": p_two,
        "n_bootstrap": int(n_bootstrap),
    }


def cohen_d_paired(
    scores_a: Sequence[float], scores_b: Sequence[float],
) -> float:
    """Paired Cohen's d (= mean diff / std of diff)."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have the same shape")
    diffs = a - b
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else 0.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 8-class smoke test mirroring the project's UNIFIED_LABEL_NAMES.
    LABELS = ["success", "no_grasp", "slip", "translation",
              "rotation", "wrong_object", "wrong_sequence", "wrong_state"]
    y_true = ["success"] * 7 + ["no_grasp"] * 2 + ["translation"]
    y_pred = ["success"] * 8 + ["no_grasp"] + ["success"]
    m = compute_classification_metrics(y_true, y_pred, LABELS)
    print(f"acc={m['accuracy']:.3f}  f1_macro={m['f1_macro']:.3f}  "
          f"f1_weighted={m['f1_weighted']:.3f}")
    print(f"warning: {m['majority_class_warning']}")

    # Bootstrap CI smoke test.
    ci = paired_bootstrap_ci([0.81, 0.83, 0.79], [0.75, 0.78, 0.74], n_bootstrap=2000)
    d = cohen_d_paired([0.81, 0.83, 0.79], [0.75, 0.78, 0.74])
    print(f"bootstrap diff={ci['mean_diff']:.3f} "
          f"CI=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
          f"p={ci['p_value_two_sided']:.3f}  cohen_d={d:.3f}")
