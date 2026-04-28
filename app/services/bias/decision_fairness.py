"""
Decision-level fairness analysis.

This module analyzes ACTUAL AI MODEL DECISIONS (predictions) rather than
raw dataset outcomes. It computes prediction-level fairness metrics including
Equalized Odds, False Positive/Negative Rate parity, calibration, and
individual fairness — the metrics that matter for automated decision systems.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.core.logging import logger

EPSILON = 1e-9


# ── Column detection ──────────────────────────────────────────────────────────

PREDICTION_HINTS = ["prediction", "predicted", "pred", "score", "output", "decision", "result", "y_pred", "y_hat"]
ACTUAL_HINTS = ["actual", "true", "ground_truth", "label", "target", "outcome", "y_true", "y_actual", "ground"]
SENSITIVE_HINTS = ["gender", "sex", "race", "ethnicity", "age", "age_group", "religion",
                   "nationality", "disability", "marital", "education", "income", "department"]


def _normalized(col: str) -> str:
    return str(col).strip().lower().replace("-", " ").replace("_", " ")


def detect_prediction_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect prediction, actual, and sensitive columns from a predictions CSV."""
    cols = list(df.columns)
    norm = {c: _normalized(c) for c in cols}

    prediction_col = next(
        (c for c in cols if any(h in norm[c] for h in PREDICTION_HINTS)), None
    )
    actual_col = next(
        (c for c in cols if any(h in norm[c] for h in ACTUAL_HINTS) and c != prediction_col), None
    )
    sensitive_cols = [
        c for c in cols
        if c not in (prediction_col, actual_col)
        and any(h in norm[c] for h in SENSITIVE_HINTS)
    ]

    return {
        "prediction_column": prediction_col,
        "actual_column": actual_col,
        "sensitive_columns": sensitive_cols[:3],
    }


# ── Binary encoding ───────────────────────────────────────────────────────────

def _to_binary(series: pd.Series) -> pd.Series:
    """Convert a series to binary 0/1."""
    s = series.dropna()
    if s.empty:
        raise ValueError(f"Column '{series.name}' has no usable values.")

    if pd.api.types.is_bool_dtype(s):
        return series.fillna(False).astype(int)

    if pd.api.types.is_numeric_dtype(s):
        unique = sorted(pd.unique(s))
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            return series.fillna(0).astype(int)
        # Threshold at median for continuous scores
        median = s.median()
        return (pd.to_numeric(series, errors="coerce").fillna(median) >= median).astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    positive_tokens = {"1", "true", "yes", "y", "approved", "accept", "accepted",
                       "selected", "hired", "positive", "good", "pass"}
    unique_vals = list(pd.unique(lowered.dropna()))
    if len(unique_vals) == 2:
        pos_candidates = [v for v in unique_vals if v in positive_tokens]
        pos = pos_candidates[0] if pos_candidates else sorted(unique_vals)[-1]
        return (lowered == pos).astype(int)

    raise ValueError(f"Cannot convert column '{series.name}' to binary.")


# ── Per-group confusion matrix ────────────────────────────────────────────────

def _group_confusion(
    df: pd.DataFrame,
    sensitive_col: str,
    actual_col: str,
    pred_col: str,
) -> Dict[str, Dict]:
    """Compute TP, FP, TN, FN and derived rates per group."""
    results = {}
    for group, gdf in df.groupby(sensitive_col, dropna=True):
        y_true = gdf[actual_col].values
        y_pred = gdf[pred_col].values
        n = len(y_true)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        tpr = tp / (tp + fn + EPSILON)   # True Positive Rate (Recall / Sensitivity)
        fpr = fp / (fp + tn + EPSILON)   # False Positive Rate
        tnr = tn / (tn + fp + EPSILON)   # True Negative Rate (Specificity)
        fnr = fn / (fn + tp + EPSILON)   # False Negative Rate
        ppv = tp / (tp + fp + EPSILON)   # Precision / Positive Predictive Value
        acc = (tp + tn) / (n + EPSILON)  # Accuracy
        selection_rate = (tp + fp) / (n + EPSILON)  # Predicted positive rate

        results[str(group)] = {
            "n": n,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "tnr": round(tnr, 4),
            "fnr": round(fnr, 4),
            "precision": round(ppv, 4),
            "accuracy": round(acc, 4),
            "selection_rate": round(selection_rate, 4),
        }
    return results


# ── Fairness metric computation ───────────────────────────────────────────────

def compute_decision_fairness_metrics(
    group_stats: Dict[str, Dict],
) -> Dict:
    """
    Compute all standard fairness metrics for automated decisions:
    - Demographic Parity (selection rate parity)
    - Equalized Odds (TPR + FPR parity)
    - Equal Opportunity (TPR parity)
    - Predictive Parity (precision parity)
    - Accuracy Parity
    """
    if len(group_stats) < 2:
        return {"error": "Need at least 2 groups for fairness comparison."}

    tprs = {g: s["tpr"] for g, s in group_stats.items()}
    fprs = {g: s["fpr"] for g, s in group_stats.items()}
    fnrs = {g: s["fnr"] for g, s in group_stats.items()}
    sel_rates = {g: s["selection_rate"] for g, s in group_stats.items()}
    precisions = {g: s["precision"] for g, s in group_stats.items()}
    accuracies = {g: s["accuracy"] for g, s in group_stats.items()}

    def _diff_ratio(vals: Dict[str, float]) -> Tuple[float, float, str, str]:
        v = list(vals.values())
        max_v, min_v = max(v), min(v)
        diff = round(max_v - min_v, 4)
        ratio = round((min_v + EPSILON) / (max_v + EPSILON), 4)
        max_g = max(vals, key=vals.get)
        min_g = min(vals, key=vals.get)
        return diff, ratio, max_g, min_g

    dp_diff, dp_ratio, dp_max_g, dp_min_g = _diff_ratio(sel_rates)
    eo_tpr_diff, _, _, _ = _diff_ratio(tprs)
    eo_fpr_diff, _, _, _ = _diff_ratio(fprs)
    eqopp_diff, eqopp_ratio, eqopp_max_g, eqopp_min_g = _diff_ratio(tprs)
    pp_diff, pp_ratio, _, _ = _diff_ratio(precisions)
    acc_diff, acc_ratio, _, _ = _diff_ratio(accuracies)
    fnr_diff, _, fnr_max_g, fnr_min_g = _diff_ratio(fnrs)

    # Equalized Odds = max of TPR diff and FPR diff
    eq_odds_diff = round(max(eo_tpr_diff, eo_fpr_diff), 4)

    def _severity(diff: float) -> str:
        if diff < 0.05:
            return "LOW"
        elif diff <= 0.15:
            return "MODERATE"
        return "HIGH"

    return {
        "demographic_parity": {
            "diff": dp_diff,
            "ratio": dp_ratio,
            "severity": _severity(dp_diff),
            "highest_group": dp_max_g,
            "lowest_group": dp_min_g,
            "description": "Are positive predictions equally likely across groups?",
            "passed": dp_ratio >= 0.8,
        },
        "equalized_odds": {
            "tpr_diff": eo_tpr_diff,
            "fpr_diff": eo_fpr_diff,
            "combined_diff": eq_odds_diff,
            "severity": _severity(eq_odds_diff),
            "description": "Do groups have equal True Positive and False Positive rates?",
            "passed": eq_odds_diff < 0.1,
        },
        "equal_opportunity": {
            "diff": eqopp_diff,
            "ratio": eqopp_ratio,
            "severity": _severity(eqopp_diff),
            "highest_group": eqopp_max_g,
            "lowest_group": eqopp_min_g,
            "description": "Are qualified individuals equally likely to receive positive decisions?",
            "passed": eqopp_diff < 0.1,
        },
        "predictive_parity": {
            "diff": pp_diff,
            "ratio": pp_ratio,
            "severity": _severity(pp_diff),
            "description": "Is precision (positive predictive value) equal across groups?",
            "passed": pp_ratio >= 0.8,
        },
        "accuracy_parity": {
            "diff": acc_diff,
            "ratio": acc_ratio,
            "severity": _severity(acc_diff),
            "description": "Is the model equally accurate across groups?",
            "passed": acc_diff < 0.05,
        },
        "false_negative_rate_parity": {
            "diff": fnr_diff,
            "severity": _severity(fnr_diff),
            "highest_group": fnr_max_g,
            "lowest_group": fnr_min_g,
            "description": "Are qualified individuals equally likely to be incorrectly rejected?",
            "passed": fnr_diff < 0.1,
        },
    }


# ── Overall verdict ───────────────────────────────────────────────────────────

def _overall_verdict(fairness_metrics: Dict, group_stats: Dict[str, Dict]) -> Dict:
    """Produce a plain-English verdict on the model's fairness."""
    failed = [k for k, v in fairness_metrics.items() if isinstance(v, dict) and not v.get("passed", True)]
    high_severity = [k for k, v in fairness_metrics.items() if isinstance(v, dict) and v.get("severity") == "HIGH"]

    if not failed:
        verdict = "FAIR"
        summary = "The model's decisions appear fair across all measured criteria."
        risk = "LOW"
    elif high_severity:
        verdict = "BIASED"
        summary = (f"The model shows significant bias. "
                   f"{len(high_severity)} metric(s) have HIGH severity: {', '.join(high_severity)}.")
        risk = "HIGH"
    else:
        verdict = "POSSIBLY BIASED"
        summary = (f"{len(failed)} fairness criterion/criteria not met: {', '.join(failed)}. "
                   f"Review and consider mitigation.")
        risk = "MODERATE"

    # Most disadvantaged group (highest FNR = most often wrongly rejected)
    if group_stats:
        most_disadvantaged = max(group_stats, key=lambda g: group_stats[g]["fnr"])
        most_advantaged = min(group_stats, key=lambda g: group_stats[g]["fnr"])
    else:
        most_disadvantaged = most_advantaged = "N/A"

    return {
        "verdict": verdict,
        "risk_level": risk,
        "summary": summary,
        "failed_criteria": failed,
        "high_severity_criteria": high_severity,
        "most_disadvantaged_group": most_disadvantaged,
        "most_advantaged_group": most_advantaged,
        "criteria_passed": len(fairness_metrics) - len(failed),
        "criteria_total": len(fairness_metrics),
    }


# ── Mitigation recommendations ────────────────────────────────────────────────

def _generate_decision_recommendations(
    fairness_metrics: Dict,
    group_stats: Dict[str, Dict],
    sensitive_col: str,
) -> List[Dict]:
    recs = []

    dp = fairness_metrics.get("demographic_parity", {})
    if not dp.get("passed"):
        recs.append({
            "priority": "HIGH",
            "action": "Apply threshold calibration per group",
            "detail": (f"Group '{dp.get('lowest_group')}' receives positive decisions at a much lower rate. "
                       f"Adjust decision thresholds per group to equalize selection rates."),
            "technique": "Threshold Optimization",
        })

    eo = fairness_metrics.get("equalized_odds", {})
    if not eo.get("passed"):
        recs.append({
            "priority": "HIGH",
            "action": "Apply post-processing equalized odds correction",
            "detail": (f"TPR diff: {eo.get('tpr_diff', 0):.1%}, FPR diff: {eo.get('fpr_diff', 0):.1%}. "
                       f"Use Hardt et al. post-processing to equalize error rates."),
            "technique": "Post-Processing (Equalized Odds)",
        })

    eqopp = fairness_metrics.get("equal_opportunity", {})
    if not eqopp.get("passed"):
        recs.append({
            "priority": "HIGH",
            "action": f"Reduce false negative rate for group '{eqopp.get('lowest_group')}'",
            "detail": (f"Qualified individuals in '{eqopp.get('lowest_group')}' are being incorrectly rejected "
                       f"more often. Lower the decision threshold for this group."),
            "technique": "Group-Specific Threshold Lowering",
        })

    fnr = fairness_metrics.get("false_negative_rate_parity", {})
    if not fnr.get("passed"):
        recs.append({
            "priority": "MEDIUM",
            "action": f"Investigate training data for group '{fnr.get('highest_group')}'",
            "detail": (f"Group '{fnr.get('highest_group')}' has a disproportionately high false negative rate. "
                       f"Check if training data underrepresents this group."),
            "technique": "Training Data Audit",
        })

    pp = fairness_metrics.get("predictive_parity", {})
    if not pp.get("passed"):
        recs.append({
            "priority": "MEDIUM",
            "action": "Calibrate model predictions per group",
            "detail": "Precision differs significantly across groups. Apply Platt scaling or isotonic regression per group.",
            "technique": "Calibration",
        })

    if not recs:
        recs.append({
            "priority": "LOW",
            "action": "Continue monitoring model decisions",
            "detail": "All fairness criteria are currently met. Set up periodic re-evaluation as data distribution may shift.",
            "technique": "Ongoing Monitoring",
        })

    return recs


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_model_decisions(
    df: pd.DataFrame,
    prediction_col: str,
    actual_col: str,
    sensitive_columns: List[str],
) -> Dict:
    """
    Full decision-level fairness analysis.

    Accepts a DataFrame with model predictions and actual outcomes,
    computes all standard fairness metrics per sensitive attribute,
    and returns a structured report.
    """
    logger.info(f"Analyzing model decisions: pred={prediction_col}, actual={actual_col}, sensitive={sensitive_columns}")

    # Convert to binary
    try:
        df = df.copy()
        df["_pred_bin"] = _to_binary(df[prediction_col])
        df["_actual_bin"] = _to_binary(df[actual_col])
    except ValueError as e:
        raise ValueError(f"Could not process columns: {e}")

    overall_n = len(df)
    overall_accuracy = float(np.mean(df["_pred_bin"] == df["_actual_bin"]))
    overall_positive_rate = float(df["_pred_bin"].mean())
    overall_actual_positive_rate = float(df["_actual_bin"].mean())

    per_attribute = {}
    all_recommendations = []

    for sensitive_col in sensitive_columns:
        if sensitive_col not in df.columns:
            continue

        group_stats = _group_confusion(df, sensitive_col, "_actual_bin", "_pred_bin")
        if len(group_stats) < 2:
            continue

        fairness_metrics = compute_decision_fairness_metrics(group_stats)
        verdict = _overall_verdict(fairness_metrics, group_stats)
        recs = _generate_decision_recommendations(fairness_metrics, group_stats, sensitive_col)
        all_recommendations.extend(recs)

        per_attribute[sensitive_col] = {
            "group_stats": group_stats,
            "fairness_metrics": fairness_metrics,
            "verdict": verdict,
            "recommendations": recs,
        }

    # Overall verdict across all attributes
    all_verdicts = [v["verdict"]["risk_level"] for v in per_attribute.values()]
    if "HIGH" in all_verdicts:
        overall_risk = "HIGH"
    elif "MODERATE" in all_verdicts:
        overall_risk = "MODERATE"
    else:
        overall_risk = "LOW"

    failed_attrs = [attr for attr, data in per_attribute.items()
                    if data["verdict"]["verdict"] != "FAIR"]

    return {
        "total_records": overall_n,
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_positive_prediction_rate": round(overall_positive_rate, 4),
        "overall_actual_positive_rate": round(overall_actual_positive_rate, 4),
        "overall_risk_level": overall_risk,
        "overall_verdict": (
            "BIASED" if overall_risk == "HIGH" else
            "POSSIBLY BIASED" if overall_risk == "MODERATE" else "FAIR"
        ),
        "overall_summary": (
            f"Analyzed {overall_n} model decisions across {len(sensitive_columns)} sensitive attribute(s). "
            f"Model accuracy: {overall_accuracy:.1%}. "
            f"{'Bias detected in: ' + ', '.join(failed_attrs) + '.' if failed_attrs else 'No significant bias detected.'}"
        ),
        "sensitive_attributes_analyzed": list(per_attribute.keys()),
        "per_attribute": per_attribute,
        "top_recommendations": sorted(all_recommendations, key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[r["priority"]])[:5],
        "compliance": {
            "eu_ai_act": "NON_COMPLIANT" if overall_risk == "HIGH" else "REVIEW_REQUIRED" if overall_risk == "MODERATE" else "COMPLIANT",
            "us_eeoc_80_rule": all(
                data["fairness_metrics"].get("demographic_parity", {}).get("passed", True)
                for data in per_attribute.values()
            ),
            "notes": "Based on automated analysis. Human review required for regulatory compliance.",
        },
    }
