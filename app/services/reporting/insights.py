from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from app.core.logging import logger


def classify_severity(dp_diff: float) -> str:
    """Classify bias severity based on demographic parity difference."""
    if dp_diff < 0.05:
        return "LOW"
    elif dp_diff <= 0.15:
        return "MODERATE"
    else:
        return "HIGH"


def compute_group_analysis(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    group_rates: Dict[str, float],
    positive_mask: pd.Series,
) -> Tuple[List[Dict], str]:
    """Compute detailed group analysis with sample sizes and reliability.
    
    Returns group analysis and a note about missing data if any.
    """
    working = dataframe[[sensitive_column]].copy()
    working["_positive"] = positive_mask.astype(int)
    grouped = working.groupby(sensitive_column, dropna=True)["_positive"].agg(["sum", "count"])
    
    # Check for missing data
    missing_count = dataframe[sensitive_column].isna().sum()
    missing_note = ""
    if missing_count > 0:
        missing_pct = (missing_count / len(dataframe)) * 100
        missing_note = (
            f"Note: {missing_count} samples ({missing_pct:.1f}%) have missing values for '{sensitive_column}' "
            f"and are excluded from group analysis to avoid misleading bias signals."
        )
    
    group_analysis = []
    for group_name, row in grouped.iterrows():
        label = str(group_name)
        total_samples = int(row["count"])
        positive_outcomes = int(row["sum"])
        selection_rate = float(group_rates.get(label, 0.0))
        
        # Mark reliability based on sample size
        reliability = "HIGH" if total_samples >= 30 else "LOW"
        
        group_analysis.append({
            "group": label,
            "total_samples": total_samples,
            "positive_outcomes": positive_outcomes,
            "selection_rate": selection_rate,
            "reliability": reliability,
        })
    
    return group_analysis, missing_note


def compute_confidence(group_analysis: List[Dict]) -> Tuple[str, str]:
    """Compute confidence level based on group sizes and balance."""
    sample_sizes = [g["total_samples"] for g in group_analysis]
    
    if not sample_sizes:
        return "LOW", "Insufficient data for confidence assessment."
    
    min_samples = min(sample_sizes)
    max_samples = max(sample_sizes)
    avg_samples = np.mean(sample_sizes)
    
    # Check if all groups have sufficient samples
    if min_samples >= 100:
        confidence = "HIGH"
        explanation = "Confidence is HIGH due to large sample sizes across all groups."
    elif min_samples >= 30:
        confidence = "MEDIUM"
        explanation = "Confidence is MEDIUM due to moderate sample sizes across groups."
    else:
        confidence = "LOW"
        explanation = f"Confidence is LOW because some groups have fewer than 30 samples (minimum: {min_samples})."
    
    # Adjust for sample imbalance
    if max_samples > 10 * min_samples:
        if confidence == "HIGH":
            confidence = "MEDIUM"
            explanation += " Sample imbalance reduces confidence."
        elif confidence == "MEDIUM":
            confidence = "LOW"
            explanation += " Significant sample imbalance reduces confidence."
    
    return confidence, explanation


def generate_recommendation(severity: str, sensitive_column: str) -> str:
    """Generate actionable recommendation based on severity."""
    if severity == "LOW":
        return "No immediate action required. Continue monitoring for changes."
    elif severity == "MODERATE":
        return (f"Monitor this disparity in '{sensitive_column}' and review feature influence. "
                "Consider investigating root causes if trend persists.")
    else:  # HIGH
        return (f"Urgent: Address the high disparity in '{sensitive_column}'. "
                "Review top bias-driving features, apply fairness constraints, "
                "and audit feature usage in decision-making processes.")


def simulate_balanced_outcome(group_rates: Dict[str, float], current_dp_diff: float) -> Dict:
    """Simulate balanced scenario where all groups have equal selection rates."""
    if not group_rates:
        return {
            "dp_diff_reduced": 0.0,
            "improvement": "0%",
            "explanation": "Cannot simulate with no group data.",
        }
    
    mean_rate = float(np.mean(list(group_rates.values())))
    balanced_dp_diff = 0.0  # All groups would have the same rate
    
    improvement = 0.0
    if current_dp_diff > 0:
        improvement = ((current_dp_diff - balanced_dp_diff) / current_dp_diff) * 100
    
    return {
        "dp_diff_reduced": float(round(balanced_dp_diff, 6)),
        "improvement": f"{improvement:.0f}%",
        "explanation": (f"If outcomes were balanced across groups (all at {mean_rate:.1%}), "
                       f"disparity could reduce from {current_dp_diff:.1%} to {balanced_dp_diff:.1%}."),
    }


def generate_proxy_explanation(sensitive_column: str, feature: str, correlation: float) -> str:
    """Generate human-readable explanation for proxy bias."""
    strength = "weakly" if correlation < 0.5 else "moderately" if correlation < 0.7 else "strongly"
    
    return (f"Feature '{feature}' is {strength} associated with {sensitive_column} "
            f"(correlation: {correlation:.2f}) and may act as a proxy, "
            f"potentially introducing indirect bias.")


def detect_intersectional_bias(
    dataframe: pd.DataFrame,
    sensitive_columns: List[str],
    positive_mask: pd.Series,
) -> List[Dict]:
    """Detect bias across combinations of sensitive attributes."""
    if len(sensitive_columns) < 2:
        return []
    
    # Create intersectional groups
    working = dataframe[sensitive_columns].copy()
    working["_positive"] = positive_mask.astype(int)
    
    # Combine columns for intersectional analysis
    working["_intersection"] = working[sensitive_columns].apply(
        lambda row: " + ".join(str(val) for val in row), axis=1
    )
    
    grouped = working.groupby("_intersection", dropna=False)["_positive"].agg(["sum", "count"])
    
    intersectional_results = []
    for group_name, row in grouped.iterrows():
        if int(row["count"]) < 30:  # Skip small groups
            continue
        
        selection_rate = float(row["sum"]) / float(row["count"])
        
        # Classify risk level
        if selection_rate < 0.3:
            risk_level = "HIGH"
        elif selection_rate < 0.5:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        intersectional_results.append({
            "group": str(group_name),
            "selection_rate": float(round(selection_rate, 6)),
            "risk_level": risk_level,
            "sample_size": int(row["count"]),
        })
    
    # Sort by risk level (HIGH first), then by selection rate (lowest first)
    risk_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    intersectional_results.sort(key=lambda x: (risk_order.get(x["risk_level"], 3), x["selection_rate"]))
    
    return intersectional_results[:3]  # Return only top 3 critical issues


def generate_distribution_context(group_rates: Dict[str, float], sensitive_column: str) -> str:
    """Add context about overall distribution to avoid misleading conclusions."""
    if not group_rates:
        return ""
    
    sorted_groups = sorted(group_rates.items(), key=lambda item: item[1], reverse=True)
    highest_group, highest_rate = sorted_groups[0]
    lowest_group, lowest_rate = sorted_groups[-1]
    
    if highest_rate > 0.7:
        return (f"Note: The '{highest_group}' group shows overall high selection rates ({highest_rate:.1%}), "
                f"which may reflect underlying population trends rather than direct bias.")
    elif lowest_rate < 0.2:
        return (f"Note: The '{lowest_group}' group shows overall low selection rates ({lowest_rate:.1%}), "
                f"which may reflect underlying population trends rather than direct bias.")
    
    return ""


def build_enhanced_explanation(
    sensitive_column: str,
    group_rates: Dict[str, float],
    dp_diff: float,
    di_ratio: float,
    group_analysis: List[Dict],
) -> str:
    """Build human-readable explanation with context."""
    if not group_rates:
        return f"No valid groups found for sensitive column '{sensitive_column}'."
    
    sorted_groups = sorted(group_rates.items(), key=lambda item: item[1], reverse=True)
    highest_group, highest_rate = sorted_groups[0]
    lowest_group, lowest_rate = sorted_groups[-1]
    
    # Calculate ratio for intuitive understanding
    ratio = highest_rate / lowest_rate if lowest_rate > 0 else float('inf')
    
    explanation = (
        f"Group '{highest_group}' has a {highest_rate:.1%} selection rate, "
        f"while '{lowest_group}' has {lowest_rate:.1%}. "
    )
    
    if ratio == float('inf') or lowest_rate == 0:
        explanation += "This represents an extreme disparity due to zero outcomes in one group."
    elif ratio < 2:
        explanation += f"This represents a {ratio:.1f}x difference in outcomes."
    elif ratio < 5:
        explanation += f"This represents a {ratio:.1f}x difference in outcomes, indicating a notable disparity."
    else:
        explanation += f"This represents a {ratio:.1f}x difference in outcomes, indicating a significant disparity."
    
    # Add sample size context
    highest_samples = next((g["total_samples"] for g in group_analysis if g["group"] == highest_group), 0)
    lowest_samples = next((g["total_samples"] for g in group_analysis if g["group"] == lowest_group), 0)
    
    if highest_samples > 0 and lowest_samples > 0:
        explanation += (f" Results are based on {highest_samples} samples for '{highest_group}' "
                      f"and {lowest_samples} for '{lowest_group}'.")
    
    return explanation

def estimate_affected_population(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    group_rates: Dict[str, float],
    mean_rate: float,
) -> Dict:
    """Estimate how many individuals are affected by bias."""
    group_counts = dataframe[sensitive_column].value_counts()
    affected_groups = []
    total_affected = 0
    
    for group, rate in group_rates.items():
        group_count = int(group_counts.get(group, 0))
        expected_positive = int(group_count * mean_rate)
        actual_positive = int(group_count * rate)
        
        if rate < mean_rate:
            disadvantaged = expected_positive - actual_positive
            total_affected += disadvantaged
            affected_groups.append({
                "group": group,
                "disadvantaged_count": int(disadvantaged),
                "total_count": group_count,
                "selection_rate": float(rate),
                "expected_rate": float(mean_rate),
            })
    
    return {
        "total_affected_individuals": int(total_affected),
        "affected_groups": affected_groups,
        "explanation": (
            f"Approximately {int(total_affected)} individuals across {len(affected_groups)} "
            f"groups may be disadvantaged by current disparities."
        ),
    }

def generate_bias_report_summary(
    fairness_metrics: Dict[str, Dict],
    sensitive_columns: List[str],
    dataframe_size: int,
) -> Dict:
    """Generate a comprehensive bias report summary for compliance."""
    high_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "HIGH"]
    moderate_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "MODERATE"]
    
    overall_risk = "HIGH" if high_severity else "MODERATE" if moderate_severity else "LOW"
    key_issue = "No significant bias detected"
    if high_severity:
        key_issue = f"High disparity in {high_severity[0]} requires immediate attention"
    elif moderate_severity:
        key_issue = f"Moderate disparity in {moderate_severity[0]} should be monitored"
    
    return {
        "overall_risk_level": overall_risk,
        "total_sensitive_attributes_analyzed": len(sensitive_columns),
        "total_records_analyzed": dataframe_size,
        "high_risk_attributes": high_severity,
        "moderate_risk_attributes": moderate_severity,
        "low_risk_attributes": [
            col for col, m in fairness_metrics.items() if m.get("severity") == "LOW"
        ],
        "compliance_status": "REQUIRES_ACTION" if high_severity else "MONITOR" if moderate_severity else "COMPLIANT",
        "key_issue": key_issue,
        "recommendation_summary": (
            f"Analysis of {dataframe_size} records across {len(sensitive_columns)} sensitive attributes "
            f"reveals {overall_risk} overall risk. "
            f"{len(high_severity)} attributes require immediate attention."
        ),
    }

def generate_structured_bias_report(
    dataframe: pd.DataFrame,
    fairness_metrics: Dict[str, Dict],
    sensitive_columns: List[str],
    target_column: str,
    bias_drivers: List[Dict],
    proxy_features: List[Dict],
    intersectional_bias: List[Dict],
    affected_population: Dict[str, Dict],
) -> Dict:
    """Generate a complete, structured bias report."""
    dataframe_size = len(dataframe)
    high_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "HIGH"]
    moderate_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "MODERATE"]
    low_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "LOW"]
    overall_risk = "HIGH" if high_severity else "MODERATE" if moderate_severity else "LOW"
    compliance_status = "Requires Action" if high_severity else "Monitor" if moderate_severity else "Safe"
    low_confidence_attrs = [col for col, m in fairness_metrics.items() if m.get("confidence") == "LOW"]

    key_issue = "No significant bias detected"
    if high_severity:
        key_issue = f"High disparity in {high_severity[0]} requires immediate attention"
    elif moderate_severity:
        key_issue = f"Moderate disparity in {moderate_severity[0]} should be monitored"

    executive_summary = (
        f"High risk detected, primarily driven by {high_severity[0]} disparity. Immediate action required."
        if high_severity else
        f"Moderate risk detected in {moderate_severity[0]}. Monitor trends and validate findings."
        if moderate_severity else
        "No significant bias detected. System shows fair outcomes across all analyzed attributes."
    )

    recommended_decision = (
        f"CRITICAL: Proceed only after mitigation of {', '.join(high_severity)} bias"
        if high_severity else
        f"WARNING: Proceed with monitoring of {', '.join(moderate_severity)} trends"
        if moderate_severity else
        "Safe to proceed with current system"
    )

    reliability_warning = (
        f"WARNING: Results may be unreliable due to small sample sizes in "
        f"{', '.join(low_confidence_attrs)}. Collect more data before making critical decisions."
        if low_confidence_attrs else ""
    )

    # ATTRIBUTE-LEVEL ANALYSIS
    attribute_analysis = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        dp_diff = metrics.get("dp_diff", 0)
        di_ratio = metrics.get("di_ratio", 1.0)
        confidence = metrics.get("confidence", "MEDIUM")
        group_analysis = metrics.get("group_analysis", [])
        low_reliability_groups = [g for g in group_analysis if g.get("reliability") == "LOW"]

        if low_reliability_groups:
            reliability_note = f"Low Reliability: {len(low_reliability_groups)} groups have <30 samples"
        else:
            reliability_note = "High Reliability: All groups have sufficient sample sizes"

        group_rates = metrics.get("group_rates", {})
        if group_rates:
            sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
            key_insight = f"{sorted_rates[0][0]} has {sorted_rates[0][1]:.1%} vs {sorted_rates[-1][0]} at {sorted_rates[-1][1]:.1%}"
        else:
            key_insight = "Insufficient data for group comparison"

        attribute_analysis.append({
            "attribute_name": col,
            "risk_level": metrics.get("severity", "LOW"),
            "dp_difference": f"{dp_diff:.1%}" if not pd.isna(dp_diff) else "Insufficient Data",
            "di_ratio": f"{di_ratio:.2f}" if not pd.isna(di_ratio) else "Insufficient Data",
            "confidence": confidence,
            "explanation": metrics.get("explanation", ""),
            "key_insight": key_insight,
            "data_reliability": reliability_note,
        })

    # GROUP DISPARITY SUMMARY
    group_disparity = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        group_rates = metrics.get("group_rates", {})
        if group_rates:
            sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
            highest_group, highest_rate = sorted_rates[0]
            lowest_group, lowest_rate = sorted_rates[-1]
            ratio = highest_rate / lowest_rate if lowest_rate > 0 else float("inf")
            group_disparity.append({
                "attribute": col,
                "highest_performing_group": highest_group,
                "lowest_performing_group": lowest_group,
                "outcome_difference": f"{ratio:.1f}x" if ratio != float("inf") else "Extreme",
            })

    # BIAS DRIVERS
    bias_drivers_explained = []
    for driver in bias_drivers:
        feature = driver.get("feature", "Unknown")
        impact = driver.get("impact", 0)
        is_proxy = any(pf.get("feature") == feature for pf in proxy_features)
        if is_proxy:
            proxy_corr = next((pf.get("correlation", 0) for pf in proxy_features if pf.get("feature") == feature), 0)
            explanation = (f"Feature '{feature}' has significant influence ({impact:.2f}) and is highly "
                           f"correlated ({proxy_corr:.2f}) with sensitive attributes, potentially encoding systemic bias.")
        else:
            explanation = (f"Feature '{feature}' has significant influence ({impact:.2f}) on predictions "
                           f"and may contribute to disparate outcomes due to historical patterns in the data.")
        bias_drivers_explained.append({
            "feature": feature,
            "impact": f"{impact:.2f}" if not pd.isna(impact) else "Insufficient Data",
            "explanation": explanation,
            "is_proxy": is_proxy,
        })

    # INTERSECTIONAL ANALYSIS
    if not intersectional_bias:
        intersectional_analysis = {
            "status": "Insufficient Data",
            "message": "Need at least 2 sensitive attributes with sufficient sample sizes (≥30 per group).",
        }
    else:
        intersectional_analysis = {
            "status": "Available",
            "top_risky_combinations": [
                {
                    "combination": item.get("group", "Unknown"),
                    "selection_rate": f"{item.get('selection_rate', 0):.1%}",
                    "risk_level": item.get("risk_level", "LOW"),
                }
                for item in intersectional_bias[:3]
            ],
        }

    # IMPACT ASSESSMENT
    impact_assessment = []
    for col, pop_data in affected_population.items():
        total_affected = pop_data.get("total_affected_individuals", 0)
        group_impacts = [
            {
                "group": g.get("group", "Unknown"),
                "disadvantaged_count": g.get("disadvantaged_count", 0),
                "total_count": g.get("total_count", 0),
            }
            for g in pop_data.get("affected_groups", [])
        ]
        impact_assessment.append({
            "attribute": col,
            "total_affected_individuals": total_affected,
            "disadvantaged_groups": group_impacts,
            "explanation": pop_data.get("explanation", ""),
        })

    # RECOMMENDATIONS
    urgent_actions = []
    monitor_actions = []
    safe_actions = []

    if high_severity:
        urgent_actions.append({
            "action": f"Apply reweighting or resampling for: {', '.join(high_severity)}",
            "reason": "High disparity detected. Implement sample reweighting to balance group representation during model training.",
            "timeline": "Within 1 week",
            "priority": "CRITICAL",
        })
        affected_proxies = [pf for pf in proxy_features if pf.get("sensitive_column") in high_severity]
        if affected_proxies:
            proxy_names = ", ".join([pf.get("feature", "unknown") for pf in affected_proxies])
            urgent_actions.append({
                "action": f"Audit and remove proxy features: {proxy_names}",
                "reason": "These features are highly correlated with sensitive attributes and may encode systemic bias.",
                "timeline": "Within 2 weeks",
                "priority": "HIGH",
            })

    if low_confidence_attrs:
        urgent_actions.append({
            "action": f"Collect more data for: {', '.join(low_confidence_attrs)}",
            "reason": "High or moderate disparity detected but confidence is LOW due to small sample sizes.",
            "timeline": "Within 1 month",
            "priority": "HIGH",
        })

    if moderate_severity:
        monitor_actions.append({
            "action": f"Monitor disparity trends for: {', '.join(moderate_severity)} over the next 30 days",
            "reason": "Moderate disparity should be tracked. Set up automated monitoring to detect if disparity increases.",
            "timeline": "Ongoing — review in 30 days",
            "priority": "MEDIUM",
        })
        monitor_actions.append({
            "action": f"Validate findings for: {', '.join(moderate_severity)} with additional datasets",
            "reason": "Cross-validation helps confirm whether disparities are consistent across datasets.",
            "timeline": "Within 2 weeks",
            "priority": "MEDIUM",
        })

    if low_severity:
        safe_actions.append({
            "action": f"Continue standard monitoring for: {', '.join(low_severity)}",
            "reason": "Low disparity indicates fair outcomes. Maintain current practices and run regular bias audits.",
            "timeline": "Quarterly review",
            "priority": "LOW",
        })

    for col in sensitive_columns:
        missing_pct = dataframe[col].isna().sum() / len(dataframe) * 100
        if missing_pct > 5:
            urgent_actions.append({
                "action": f"Handle missing values in '{col}' using imputation",
                "reason": f"{missing_pct:.1f}% missing data may bias results. Use median/mode imputation or create an 'unknown' category.",
                "timeline": "Within 1 week",
                "priority": "HIGH",
            })

    # DATA ISSUES
    data_issues = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        di_ratio = metrics.get("di_ratio", 1.0)
        group_analysis = metrics.get("group_analysis", [])
        if pd.isna(di_ratio) or di_ratio == 0 or di_ratio > 10:
            data_issues.append({
                "attribute": col,
                "issue": "Extreme Disparate Impact Ratio",
                "explanation": f"DI ratio of {di_ratio} indicates extreme imbalance, possibly due to very small group sizes.",
            })
        low_sample_groups = [g for g in group_analysis if g.get("total_samples", 0) < 30]
        if low_sample_groups:
            data_issues.append({
                "attribute": col,
                "issue": "Low Sample Size Warning",
                "explanation": f"{len(low_sample_groups)} groups have fewer than 30 samples, which may lead to unreliable conclusions.",
            })

    return {
        "overall_summary": {
            "risk_level": overall_risk,
            "records_analyzed": dataframe_size,
            "sensitive_attributes": len(sensitive_columns),
            "compliance_status": compliance_status,
            "key_issue": key_issue,
            "executive_summary": executive_summary,
            "recommended_decision": recommended_decision,
            "reliability_warning": reliability_warning,
        },
        "attribute_level_analysis": attribute_analysis,
        "group_disparity_summary": group_disparity,
        "bias_drivers": bias_drivers_explained,
        "intersectional_analysis": intersectional_analysis,
        "impact_assessment": impact_assessment,
        "recommendations": {
            "urgent_actions": urgent_actions,
            "monitor_actions": monitor_actions,
            "safe_actions": safe_actions,
        },
        "data_issues": data_issues,
    }
