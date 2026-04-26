"""Decision intelligence, monitoring, and gating services."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_improvement_percent(raw_value: Any) -> float:
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().replace("%", "")
        try:
            return float(normalized)
        except ValueError:
            return 0.0
    return 0.0


def _confidence_penalty(confidence: str) -> float:
    token = str(confidence or "").strip().upper()
    if token == "HIGH":
        return 0.0
    if token == "MEDIUM":
        return 10.0
    return 20.0


def _build_metric_aggregation(analysis_payload: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    fairness_metrics = analysis_payload.get("fairness_metrics") or {}
    affected_population = analysis_payload.get("affected_population") or {}

    if not fairness_metrics:
        empty_agg = {"dp": 0.0, "di_shortfall": 0.0, "confidence_penalty": 0.0, "impact": 0.0}
        return empty_agg, []

    dp_values: List[float] = []
    di_shortfalls: List[float] = []
    confidence_penalties: List[float] = []
    impact_values: List[float] = []
    per_attribute_scores: List[Dict[str, Any]] = []

    for attribute, metric in fairness_metrics.items():
        dp_diff = float(metric.get("dp_diff", 0.0))
        di_ratio = float(metric.get("di_ratio", 1.0))
        confidence = str(metric.get("confidence", "LOW"))

        dp_component = _clamp(dp_diff * 200.0, 0.0, 100.0)
        di_component = _clamp((1.0 - di_ratio) * 100.0, 0.0, 100.0)
        conf_component = _confidence_penalty(confidence)

        affected = affected_population.get(attribute, {}).get("total_affected_individuals", 0)
        impact_component = _clamp(float(affected) / 5.0, 0.0, 100.0)

        aggregate_score = (
            (0.40 * dp_component)
            + (0.35 * di_component)
            + (0.15 * conf_component)
            + (0.10 * impact_component)
        )

        dp_values.append(dp_component)
        di_shortfalls.append(di_component)
        confidence_penalties.append(conf_component)
        impact_values.append(impact_component)

        if aggregate_score >= 75:
            priority_level = "HIGH"
        elif aggregate_score >= 50:
            priority_level = "MEDIUM"
        else:
            priority_level = "LOW"

        per_attribute_scores.append(
            {
                "attribute": attribute,
                "priority_score": round(aggregate_score, 2),
                "priority_level": priority_level,
                "rationale": (
                    f"DP={dp_diff:.3f}, DI={di_ratio:.3f}, confidence={confidence}, "
                    f"estimated impact={int(affected)} individuals"
                ),
            }
        )

    aggregation = {
        "dp": round(sum(dp_values) / len(dp_values), 2),
        "di_shortfall": round(sum(di_shortfalls) / len(di_shortfalls), 2),
        "confidence_penalty": round(sum(confidence_penalties) / len(confidence_penalties), 2),
        "impact": round(sum(impact_values) / len(impact_values), 2),
    }

    per_attribute_scores.sort(key=lambda item: item["priority_score"], reverse=True)
    return aggregation, per_attribute_scores


def compute_unified_bias_risk_score(analysis_payload: Dict[str, Any]) -> Tuple[int, Dict[str, float], List[Dict[str, Any]]]:
    """Compute a 0-100 unified risk score from fairness and impact signals."""
    aggregation, prioritization = _build_metric_aggregation(analysis_payload)
    score = (
        (0.40 * aggregation["dp"])
        + (0.35 * aggregation["di_shortfall"])
        + (0.15 * aggregation["confidence_penalty"])
        + (0.10 * aggregation["impact"])
    )
    unified_score = int(round(_clamp(score, 0.0, 100.0)))
    return unified_score, aggregation, prioritization


def detect_bias_drift(current_score: float, historical_scores: List[float], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Detect bias drift against historical risk scores."""
    drift_abs_threshold = float(thresholds.get("drift_abs", 10.0))
    drift_pct_threshold = float(thresholds.get("drift_pct", 20.0))

    if not historical_scores:
        return {
            "detected": False,
            "delta": 0.0,
            "percent_change": 0.0,
            "baseline_score": current_score,
            "current_score": current_score,
            "explanation": "No historical baseline provided; drift detection skipped.",
        }

    baseline = sum(historical_scores) / len(historical_scores)
    delta = current_score - baseline

    if baseline > 0:
        pct_change = (delta / baseline) * 100.0
    else:
        pct_change = 100.0 if current_score > 0 else 0.0

    detected = abs(delta) >= drift_abs_threshold or abs(pct_change) >= drift_pct_threshold

    if detected:
        explanation = (
            f"Bias drift detected: current score {current_score:.1f} vs baseline {baseline:.1f} "
            f"(delta {delta:+.1f}, {pct_change:+.1f}%)."
        )
    else:
        explanation = (
            f"No significant drift: current score {current_score:.1f} vs baseline {baseline:.1f} "
            f"(delta {delta:+.1f}, {pct_change:+.1f}%)."
        )

    return {
        "detected": detected,
        "delta": round(delta, 2),
        "percent_change": round(pct_change, 2),
        "baseline_score": round(baseline, 2),
        "current_score": round(current_score, 2),
        "explanation": explanation,
    }


def build_threshold_alerts(
    score: float,
    drift: Dict[str, Any],
    thresholds: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Generate threshold-based alerts for monitor flows."""
    block_threshold = float(thresholds.get("block", 75.0))
    flag_threshold = float(thresholds.get("flag", 50.0))

    alerts: List[Dict[str, Any]] = []

    if score >= block_threshold:
        alerts.append(
            {
                "level": "CRITICAL",
                "message": "Unified bias risk score exceeded block threshold.",
                "threshold": block_threshold,
                "current_value": score,
            }
        )
    elif score >= flag_threshold:
        alerts.append(
            {
                "level": "WARNING",
                "message": "Unified bias risk score exceeded flag threshold.",
                "threshold": flag_threshold,
                "current_value": score,
            }
        )

    if drift.get("detected"):
        alerts.append(
            {
                "level": "WARNING",
                "message": "Bias drift was detected against historical baseline.",
                "threshold": float(thresholds.get("drift_abs", 10.0)),
                "current_value": abs(float(drift.get("delta", 0.0))),
            }
        )

    if not alerts:
        alerts.append(
            {
                "level": "INFO",
                "message": "No alert thresholds crossed.",
                "threshold": flag_threshold,
                "current_value": score,
            }
        )

    return alerts


def build_risk_heatmap(prioritization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a lightweight heatmap model for frontend rendering."""
    heatmap: List[Dict[str, Any]] = []
    for item in prioritization:
        score = float(item.get("priority_score", 0.0))
        if score >= 75:
            risk_level = "HIGH"
        elif score >= 50:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        heatmap.append(
            {
                "attribute": str(item.get("attribute", "unknown")),
                "score": round(score, 2),
                "risk_level": risk_level,
            }
        )
    return heatmap


def build_data_intelligence(analysis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate data quality and sampling insights for pre-analysis correction."""
    insights: List[Dict[str, Any]] = []
    notes = analysis_payload.get("notes") or []
    fairness_metrics = analysis_payload.get("fairness_metrics") or {}

    # Missing data bias detection
    missing_notes = [note for note in notes if "missing" in str(note).lower()]
    for note in missing_notes:
        insights.append(
            {
                "category": "missing_data_bias",
                "severity": "MEDIUM",
                "finding": str(note),
                "suggestion": "Apply imputation or introduce explicit unknown-category handling.",
            }
        )

    # Low sample size detection and skew detection
    for attribute, metric in fairness_metrics.items():
        group_analysis = metric.get("group_analysis") or []
        low_sample_groups = [g for g in group_analysis if int(g.get("total_samples", 0)) < 30]
        if low_sample_groups:
            insights.append(
                {
                    "category": "low_sample_size",
                    "severity": "HIGH",
                    "finding": (
                        f"{len(low_sample_groups)} groups under 30 samples in '{attribute}' "
                        "reduce reliability."
                    ),
                    "suggestion": "Collect additional samples or aggregate sparse categories before decisions.",
                }
            )

        group_rates = metric.get("group_rates") or {}
        if len(group_rates) >= 2:
            max_rate = max(group_rates.values())
            min_rate = min(group_rates.values())
            if min_rate > 0:
                ratio = max_rate / min_rate
            else:
                ratio = float("inf")
            if ratio >= 2:
                insights.append(
                    {
                        "category": "skew_detection",
                        "severity": "MEDIUM" if ratio < 3 else "HIGH",
                        "finding": (
                            f"Selection-rate skew in '{attribute}' is {ratio:.2f}x between highest and lowest groups."
                        ),
                        "suggestion": "Review sampling strategy and apply reweighting before deployment.",
                    }
                )

    if not insights:
        insights.append(
            {
                "category": "data_quality",
                "severity": "LOW",
                "finding": "No major data intelligence issues detected from supplied analysis payload.",
                "suggestion": "Continue periodic monitoring for drift and sparse subgroup effects.",
            }
        )

    return insights


def build_mitigation_preview(analysis_payload: Dict[str, Any], prioritization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create mitigation impact previews for top-priority attributes."""
    fairness_metrics = analysis_payload.get("fairness_metrics") or {}
    preview: List[Dict[str, Any]] = []

    top_items = prioritization[:3]
    for item in top_items:
        attribute = str(item.get("attribute", "unknown"))
        metric = fairness_metrics.get(attribute, {})
        sim = metric.get("simulation") or {}
        gain_raw = sim.get("improvement", "0%")
        gain_value = _parse_improvement_percent(gain_raw)

        for method, factor, notes in [
            ("auto_reweighting", 1.0, "Balances subgroup contribution in model training."),
            ("auto_resampling", 0.8, "Expands underrepresented groups to reduce sparsity bias."),
            ("threshold_adjustment", 0.6, "Aligns acceptance thresholds across groups for parity."),
        ]:
            expected = _clamp(gain_value * factor, 0.0, 100.0)
            preview.append(
                {
                    "attribute": attribute,
                    "method": method,
                    "expected_risk_reduction": round(expected, 2),
                    "expected_fairness_gain": f"{expected:.0f}% estimated disparity reduction",
                    "notes": notes,
                }
            )

    return preview


def build_explainability(analysis_payload: Dict[str, Any], prioritization: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """Build human-readable explanations, impacted groups, and consequence narratives."""
    fairness_metrics = analysis_payload.get("fairness_metrics") or {}
    affected_population = analysis_payload.get("affected_population") or {}

    explanations: List[str] = []
    impacted_groups: List[str] = []
    consequences: List[str] = []

    for item in prioritization[:3]:
        attribute = str(item.get("attribute", "unknown"))
        metric = fairness_metrics.get(attribute, {})
        explanation = str(metric.get("explanation", f"No explanation available for '{attribute}'."))
        explanations.append(explanation)

        affected = affected_population.get(attribute, {})
        for group_item in affected.get("affected_groups", []):
            group_name = str(group_item.get("group", "unknown"))
            impacted_groups.append(f"{attribute}:{group_name}")

        consequences.append(
            f"If unmitigated, decisions in '{attribute}' may trigger disproportionate denial outcomes and compliance risk."
        )

    unique_groups = sorted(set(impacted_groups))
    if not unique_groups:
        unique_groups = ["No materially impacted groups identified from provided payload."]

    if not explanations:
        explanations = ["No detailed fairness explanations were found in the provided payload."]

    return explanations, unique_groups, consequences


def build_audit_report(
    scenario: str,
    score: int,
    alerts: List[Dict[str, Any]],
    drift: Dict[str, Any],
    prioritization: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a lightweight compliance report for audit and governance tracking."""
    if score >= 75:
        compliance_status = "REQUIRES_ACTION"
    elif score >= 50:
        compliance_status = "MONITOR"
    else:
        compliance_status = "COMPLIANT"

    return {
        "report_id": f"audit-{uuid4().hex[:12]}",
        "generated_at": _now_iso(),
        "scenario": scenario,
        "overall_risk_level": "HIGH" if score >= 75 else "MODERATE" if score >= 50 else "LOW",
        "compliance_status": compliance_status,
        "drift_detected": bool(drift.get("detected", False)),
        "alert_count": len(alerts),
        "top_priorities": prioritization[:3],
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def append_tracking_log(event_type: str, payload: Dict[str, Any]) -> str:
    """Persist decision/mitigation monitoring logs for governance tracking."""
    logs_dir = _repo_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_id = f"{event_type}-{uuid4().hex[:12]}"
    record = {
        "log_id": log_id,
        "event_type": event_type,
        "timestamp": _now_iso(),
        "payload": payload,
    }

    log_file = logs_dir / f"{event_type}.jsonl"
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return log_id


def evaluate_gate(
    decision_id: str,
    scenario: str,
    score: int,
    flag_threshold: int,
    block_threshold: int,
    impacted_groups: List[str],
    risk_consequences: List[str],
    auto_mitigation: bool,
) -> Dict[str, Any]:
    """Apply bias blocking/flagging logic and fairness approval rules."""
    reasons: List[str] = []
    mitigation_actions: List[str] = []

    if score >= block_threshold:
        status = "BLOCKED"
        fairness_approval_required = True
        fairness_approved = False
        reasons.append("Bias risk exceeded block threshold.")
    elif score >= flag_threshold:
        status = "FLAGGED"
        fairness_approval_required = True
        fairness_approved = False
        reasons.append("Bias risk exceeded flag threshold.")
    else:
        status = "APPROVED"
        fairness_approval_required = False
        fairness_approved = True
        reasons.append("Bias risk is below gating thresholds.")

    if auto_mitigation and status != "APPROVED":
        mitigation_actions.extend(
            [
                "Run auto_reweighting on affected attributes before execution.",
                "Run auto_resampling for sparse impacted groups.",
                "Apply threshold_adjustment and re-evaluate fairness score.",
            ]
        )

    fairness_certificate = {
        "certificate_type": "Fairness Pre-Decision Check (Non-Legal)",
        "decision_id": decision_id,
        "scenario": scenario,
        "issued_at": _now_iso(),
        "status": status,
        "risk_score": score,
        "approval_required": fairness_approval_required,
    }

    return {
        "status": status,
        "fairness_approval_required": fairness_approval_required,
        "fairness_approved": fairness_approved,
        "reasons": reasons,
        "impacted_groups": impacted_groups,
        "risk_consequences": risk_consequences,
        "mitigation_actions": mitigation_actions,
        "fairness_certificate": fairness_certificate,
    }
