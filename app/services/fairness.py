from typing import Dict, List

import pandas as pd

from app.services.insights import (
    compute_group_analysis,
    classify_severity,
    compute_confidence,
    generate_recommendation,
    build_enhanced_explanation,
    detect_intersectional_bias,
    generate_distribution_context,
    generate_proxy_explanation,
    simulate_balanced_outcome,
)
from app.services.metrics import (
    compute_fairness_metrics,
    compute_group_selection_rates,
    encode_positive_mask,
)
from app.services.mitigation import (
    estimate_affected_population,
    generate_bias_report_summary,
    generate_data_preprocessing_steps,
    suggest_feature_removal,
    generate_structured_bias_report,
)
from app.services.patterns import detect_bias_drivers, detect_proxy_features


def _build_explanation(sensitive_column: str, group_rates: Dict[str, float], dp_diff: float) -> str:
    if not group_rates:
        return f"No valid groups found for sensitive column '{sensitive_column}'."

    sorted_groups = sorted(group_rates.items(), key=lambda item: item[1], reverse=True)
    highest_group, highest_rate = sorted_groups[0]
    lowest_group, lowest_rate = sorted_groups[-1]

    return (
        f"Within '{sensitive_column}', group '{highest_group}' has a {highest_rate:.1%} selection rate, "
        f"while group '{lowest_group}' has {lowest_rate:.1%}. "
        f"This indicates a disparity of {dp_diff:.1%}, suggesting potential bias."
    )


def _build_summary(target_column: str, sensitive_columns: List[str], fairness_metrics: Dict[str, Dict]) -> str:
    if not fairness_metrics:
        return f"Bias analysis completed for target '{target_column}', but no fairness metrics were produced."

    strongest_column, strongest_metrics = max(
        fairness_metrics.items(),
        key=lambda item: item[1]["dp_diff"],
    )
    return (
        f"Analyzed target '{target_column}' across {len(sensitive_columns)} sensitive column(s). "
        f"The largest observed disparity was in '{strongest_column}' with demographic parity difference "
        f"of {strongest_metrics['dp_diff']:.1%} and disparate impact ratio of {strongest_metrics['di_ratio']:.3f}."
    )


async def analyze_dataset_bias(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_columns: List[str],
) -> Dict:
    positive_mask, _ = encode_positive_mask(dataframe[target_column])

    fairness_results: Dict[str, Dict] = {}
    notes = []
    
    for sensitive_column in sensitive_columns:
        group_rates = compute_group_selection_rates(dataframe, sensitive_column, positive_mask)
        metrics = compute_fairness_metrics(group_rates)
        
        # Enhanced analysis
        severity = classify_severity(metrics["dp_diff"])
        group_analysis, missing_note = compute_group_analysis(dataframe, sensitive_column, group_rates, positive_mask)
        
        # Add missing data note if any
        if missing_note:
            notes.append(missing_note)
        
        confidence, confidence_explanation = compute_confidence(group_analysis)
        recommendation = generate_recommendation(severity, sensitive_column)
        simulation = simulate_balanced_outcome(group_rates, metrics["dp_diff"])
        
        # Add distribution context note
        dist_context = generate_distribution_context(group_rates, sensitive_column)
        if dist_context:
            notes.append(dist_context)
        
        fairness_results[sensitive_column] = {
            **metrics,
            "group_rates": group_rates,
            "explanation": await build_enhanced_explanation(
                sensitive_column=sensitive_column,
                group_rates=group_rates,
                dp_diff=metrics["dp_diff"],
                di_ratio=metrics["di_ratio"],
                group_analysis=group_analysis,
            ),
            "severity": severity,
            "group_analysis": group_analysis,
            "confidence": confidence,
            "confidence_explanation": confidence_explanation,
            "recommendation": recommendation,
            "simulation": simulation,
        }

    potential_bias_detected = any(
        metric["dp_diff"] > 0.1 or metric["di_ratio"] < 0.8
        for metric in fairness_results.values()
    )

    # Enhanced proxy features with explanations
    proxy_features_raw = detect_proxy_features(dataframe, target_column, sensitive_columns)
    proxy_features = [
        {
            **proxy,
            "explanation": generate_proxy_explanation(
                proxy["sensitive_column"],
                proxy["feature"],
                proxy["correlation"]
            )
        }
        for proxy in proxy_features_raw
    ]

    # Intersectional bias detection
    intersectional_bias = detect_intersectional_bias(dataframe, sensitive_columns, positive_mask)

    # Affected population estimation
    affected_population = {}
    for sensitive_column in sensitive_columns:
        if sensitive_column in fairness_results:
            group_rates = fairness_results[sensitive_column]["group_rates"]
            mean_rate = sum(group_rates.values()) / len(group_rates)
            affected_population[sensitive_column] = estimate_affected_population(
                dataframe, sensitive_column, group_rates, mean_rate
            )

    # Preprocessing recommendations
    preprocessing_steps = generate_data_preprocessing_steps(
        dataframe, sensitive_columns, fairness_results
    )

    # Feature removal suggestions
    feature_removals = suggest_feature_removal(proxy_features)

    # Bias report summary for compliance
    bias_report_summary = generate_bias_report_summary(
        fairness_results, sensitive_columns, len(dataframe)
    )

    # Structured bias report following the specified format
    structured_report = await generate_structured_bias_report(
        dataframe=dataframe,
        fairness_metrics=fairness_results,
        sensitive_columns=sensitive_columns,
        target_column=target_column,
        bias_drivers=detect_bias_drivers(dataframe, target_column),
        proxy_features=proxy_features,
        intersectional_bias=intersectional_bias,
        affected_population=affected_population,
    )

    return {
        "summary": _build_summary(target_column, sensitive_columns, fairness_results),
        "detected_target": target_column,
        "detected_sensitive_columns": sensitive_columns,
        "potential_bias_detected": potential_bias_detected,
        "fairness_metrics": fairness_results,
        "bias_drivers": detect_bias_drivers(dataframe, target_column),
        "proxy_features": proxy_features,
        "intersectional_bias": intersectional_bias,
        "notes": notes,
        "affected_population": affected_population,
        "preprocessing_steps": preprocessing_steps,
        "feature_removals": feature_removals,
        "bias_report_summary": bias_report_summary,
        "structured_bias_report": structured_report,
    }
