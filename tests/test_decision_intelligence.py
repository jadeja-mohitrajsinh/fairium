from app.services.decision_intelligence import (
    build_threshold_alerts,
    compute_unified_bias_risk_score,
    detect_bias_drift,
    evaluate_gate,
)


def _sample_analysis_payload() -> dict:
    return {
        "fairness_metrics": {
            "gender": {
                "dp_diff": 0.20,
                "di_ratio": 0.60,
                "confidence": "MEDIUM",
                "group_rates": {"male": 0.8, "female": 0.6},
                "simulation": {"improvement": "40%"},
                "explanation": "Gender disparity is visible.",
            },
            "race": {
                "dp_diff": 0.12,
                "di_ratio": 0.72,
                "confidence": "HIGH",
                "group_rates": {"group_a": 0.7, "group_b": 0.58},
                "simulation": {"improvement": "28%"},
                "explanation": "Race disparity is moderate.",
            },
        },
        "affected_population": {
            "gender": {
                "total_affected_individuals": 75,
                "affected_groups": [{"group": "female", "disadvantaged_count": 75}],
            },
            "race": {
                "total_affected_individuals": 48,
                "affected_groups": [{"group": "group_b", "disadvantaged_count": 48}],
            },
        },
        "notes": ["4% missing values in sensitive columns"],
    }


def test_unified_bias_risk_score_is_bounded_and_ranked() -> None:
    score, aggregation, prioritization = compute_unified_bias_risk_score(_sample_analysis_payload())

    assert 0 <= score <= 100
    assert set(aggregation.keys()) == {"dp", "di_shortfall", "confidence_penalty", "impact"}
    assert len(prioritization) == 2
    assert prioritization[0]["priority_score"] >= prioritization[1]["priority_score"]


def test_detect_bias_drift_detects_significant_change() -> None:
    drift = detect_bias_drift(
        current_score=72.0,
        historical_scores=[45.0, 48.0, 50.0, 52.0],
        thresholds={"drift_abs": 10.0, "drift_pct": 20.0},
    )

    assert drift["detected"] is True
    assert drift["delta"] > 0
    assert drift["percent_change"] > 20


def test_detect_bias_drift_with_no_history_is_safe() -> None:
    drift = detect_bias_drift(
        current_score=40.0,
        historical_scores=[],
        thresholds={"drift_abs": 10.0, "drift_pct": 20.0},
    )

    assert drift["detected"] is False
    assert drift["baseline_score"] == 40.0
    assert "skipped" in drift["explanation"].lower()


def test_threshold_alerts_include_warning_when_flagged() -> None:
    drift = {
        "detected": False,
        "delta": 0.0,
    }
    alerts = build_threshold_alerts(
        score=55.0,
        drift=drift,
        thresholds={"flag": 50.0, "block": 75.0, "drift_abs": 10.0},
    )

    assert any(alert["level"] == "WARNING" for alert in alerts)


def test_gate_policy_blocked_path_requires_approval() -> None:
    outcome = evaluate_gate(
        decision_id="loan-100",
        scenario="lending",
        score=82,
        flag_threshold=50,
        block_threshold=75,
        impacted_groups=["race:group_b"],
        risk_consequences=["Potential regulatory non-compliance"],
        auto_mitigation=True,
    )

    assert outcome["status"] == "BLOCKED"
    assert outcome["fairness_approval_required"] is True
    assert outcome["fairness_approved"] is False
    assert len(outcome["mitigation_actions"]) >= 1


def test_gate_policy_approved_path_is_clear() -> None:
    outcome = evaluate_gate(
        decision_id="hire-100",
        scenario="hiring",
        score=35,
        flag_threshold=50,
        block_threshold=75,
        impacted_groups=[],
        risk_consequences=[],
        auto_mitigation=True,
    )

    assert outcome["status"] == "APPROVED"
    assert outcome["fairness_approval_required"] is False
    assert outcome["fairness_approved"] is True
