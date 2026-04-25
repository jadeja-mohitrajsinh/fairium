"""Bias mitigation and remediation strategies."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def compute_reweighting_weights(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    target_column: str,
) -> Tuple[pd.Series, str]:
    """
    Compute sample weights to balance representation across sensitive groups.
    
    Returns weights and explanation.
    """
    # Get group sizes
    group_counts = dataframe[sensitive_column].value_counts()
    total_samples = len(dataframe)
    
    # Target weight for equal representation
    target_weight = total_samples / len(group_counts)
    
    # Compute weights
    weights = dataframe[sensitive_column].map(lambda x: target_weight / group_counts[x])
    
    explanation = (
        f"Reweighting assigns higher weights to underrepresented groups in '{sensitive_column}' "
        f"to balance their influence during model training. "
        f"Largest group has {group_counts.max()} samples, smallest has {group_counts.min()} samples."
    )
    
    return weights, explanation


def suggest_feature_removal(
    proxy_features: List[Dict],
    correlation_threshold: float = 0.7,
) -> List[Dict]:
    """
    Suggest removing high-correlation proxy features.
    
    Returns list of features to remove with rationale.
    """
    removals = []
    
    for proxy in proxy_features:
        if proxy["correlation"] >= correlation_threshold:
            removals.append({
                "feature": proxy["feature"],
                "sensitive_column": proxy["sensitive_column"],
                "correlation": proxy["correlation"],
                "rationale": (
                    f"Feature '{proxy['feature']}' has high correlation ({proxy['correlation']:.2f}) "
                    f"with '{proxy['sensitive_column']}' and should be removed or transformed "
                    f"to prevent indirect bias."
                ),
                "priority": "HIGH" if proxy["correlation"] >= 0.8 else "MEDIUM",
            })
    
    return removals


def generate_data_preprocessing_steps(
    dataframe: pd.DataFrame,
    sensitive_columns: List[str],
    fairness_metrics: Dict[str, Dict],
) -> List[Dict]:
    """
    Generate data preprocessing recommendations to reduce bias.
    """
    steps = []
    
    # Check for missing values in sensitive columns
    for col in sensitive_columns:
        missing_pct = dataframe[col].isna().sum() / len(dataframe) * 100
        if missing_pct > 0:
            steps.append({
                "type": "missing_values",
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "recommendation": (
                    f"Impute missing values in '{col}' using median/mode or "
                    f"create a separate 'unknown' category to avoid bias from exclusion."
                ),
                "priority": "MEDIUM" if missing_pct < 5 else "HIGH",
            })
    
    # Check for class imbalance in target
    target_col = fairness_metrics.get("detected_target", "")
    if target_col and target_col in dataframe.columns:
        value_counts = dataframe[target_col].value_counts(normalize=True)
        if len(value_counts) == 2:
            imbalance_ratio = value_counts.max() / value_counts.min()
            if imbalance_ratio > 3:
                steps.append({
                    "type": "class_imbalance",
                    "column": target_col,
                    "issue": f"Class imbalance ratio: {imbalance_ratio:.1f}x",
                    "recommendation": (
                        f"Consider oversampling the minority class or using "
                        f"class weights to prevent model bias toward the majority class."
                    ),
                    "priority": "MEDIUM",
                })
    
    # Suggest reweighting for high disparity
    for col, metrics in fairness_metrics.items():
        if col in sensitive_columns and metrics.get("severity") == "HIGH":
            steps.append({
                "type": "reweighting",
                "column": col,
                "issue": f"High disparity detected (DP diff: {metrics['dp_diff']:.1%})",
                "recommendation": (
                    f"Apply sample reweighting to balance group representation "
                    f"for '{col}' during model training."
                ),
                "priority": "HIGH",
            })
    
    return steps


def estimate_affected_population(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    group_rates: Dict[str, float],
    mean_rate: float,
) -> Dict:
    """
    Estimate how many individuals are affected by bias.
    """
    group_counts = dataframe[sensitive_column].value_counts()
    
    affected_groups = []
    total_affected = 0
    
    for group, rate in group_rates.items():
        group_count = int(group_counts.get(group, 0))
        
        # Estimate how many would have positive outcome if rate were mean
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
    """
    Generate a comprehensive bias report summary for compliance.
    """
    high_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "HIGH"]
    moderate_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "MODERATE"]
    
    overall_risk = "HIGH" if high_severity else "MODERATE" if moderate_severity else "LOW"
    
    # Identify key issue (most critical biased attribute)
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
    """
    Generate a complete, structured bias report following the specified format.
    """
    dataframe_size = len(dataframe)
    
    # 1. OVERALL SUMMARY
    high_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "HIGH"]
    moderate_severity = [col for col, m in fairness_metrics.items() if m.get("severity") == "MODERATE"]
    overall_risk = "HIGH" if high_severity else "MODERATE" if moderate_severity else "LOW"
    compliance_status = "Requires Action" if high_severity else "Monitor" if moderate_severity else "Safe"
    
    key_issue = "No significant bias detected"
    if high_severity:
        key_issue = f"High disparity in {high_severity[0]} requires immediate attention"
    elif moderate_severity:
        key_issue = f"Moderate disparity in {moderate_severity[0]} should be monitored"
    
    # 2. ATTRIBUTE-LEVEL ANALYSIS
    attribute_analysis = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        
        dp_diff = metrics.get("dp_diff", 0)
        di_ratio = metrics.get("di_ratio", 1.0)
        severity = metrics.get("severity", "LOW")
        confidence = metrics.get("confidence", "MEDIUM")
        group_analysis = metrics.get("group_analysis", [])
        
        # Data reliability check
        low_reliability_groups = [g for g in group_analysis if g.get("reliability") == "LOW"]
        if low_reliability_groups:
            reliability_note = f"Low Reliability: {len(low_reliability_groups)} groups have <100 samples"
            if confidence == "HIGH":
                confidence = "MEDIUM"
            # Add explanation for confidence vs risk mismatch
            if severity == "HIGH" and confidence == "LOW":
                reliability_note += ". Note: High disparity detected but LOW confidence due to small sample sizes. Collect more data to validate findings."
            elif severity == "MODERATE" and confidence == "LOW":
                reliability_note += ". Note: Moderate disparity with LOW confidence. Results may not be statistically significant."
        else:
            reliability_note = "High Reliability: All groups have sufficient sample sizes"
        
        # Key insight
        group_rates = metrics.get("group_rates", {})
        if group_rates:
            sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
            highest = sorted_rates[0]
            lowest = sorted_rates[-1]
            key_insight = f"{highest[0]} has {highest[1]:.1%} selection rate vs {lowest[0]} at {lowest[1]:.1%}"
        else:
            key_insight = "Insufficient data for group comparison"
        
        attribute_analysis.append({
            "attribute_name": col,
            "risk_level": severity,
            "dp_difference": f"{dp_diff:.1%}" if not pd.isna(dp_diff) else "Insufficient Data",
            "di_ratio": f"{di_ratio:.2f}" if not pd.isna(di_ratio) and di_ratio > 0 else "Insufficient Data",
            "confidence": confidence,
            "explanation": metrics.get("explanation", "No explanation available"),
            "key_insight": key_insight,
            "data_reliability": reliability_note,
        })
    
    # 3. GROUP DISPARITY SUMMARY
    group_disparity = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        
        group_rates = metrics.get("group_rates", {})
        if group_rates:
            sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
            highest_group, highest_rate = sorted_rates[0]
            lowest_group, lowest_rate = sorted_rates[-1]
            ratio = highest_rate / lowest_rate if lowest_rate > 0 else float('inf')
            
            group_disparity.append({
                "attribute": col,
                "highest_performing_group": highest_group,
                "lowest_performing_group": lowest_group,
                "outcome_difference": f"{ratio:.1f}×" if ratio != float('inf') else "Extreme",
            })
    
    # 4. BIAS DRIVERS
    bias_drivers_explained = []
    for driver in bias_drivers:
        feature = driver.get("feature", "Unknown")
        impact = driver.get("impact", 0)
        
        # Generate context-aware explanation for why this feature causes bias
        explanation = f"Feature '{feature}' has significant influence ({impact:.2f}) on predictions"
        
        # Check if this feature might be a proxy
        is_proxy = any(pf.get("feature") == feature for pf in proxy_features)
        if is_proxy:
            proxy_corr = next((pf.get("correlation", 0) for pf in proxy_features if pf.get("feature") == feature), 0)
            explanation += f" and is highly correlated ({proxy_corr:.2f}) with sensitive attributes, potentially encoding systemic bias."
        else:
            explanation += " and may contribute to disparate outcomes across groups due to historical patterns in the data."
        
        bias_drivers_explained.append({
            "feature": feature,
            "impact": f"{impact:.2f}" if not pd.isna(impact) else "Insufficient Data",
            "explanation": explanation,
            "is_proxy": is_proxy,
        })
    
    # 5. INTERSECTIONAL ANALYSIS
    if not intersectional_bias or len(intersectional_bias) == 0:
        intersectional_analysis = {
            "status": "Insufficient Data",
            "message": "Insufficient Data for intersectional analysis - need at least 2 sensitive attributes with sufficient sample sizes"
        }
    else:
        top_risky = intersectional_bias[:3]
        intersectional_analysis = {
            "status": "Available",
            "top_risky_combinations": [
                {
                    "combination": item.get("group", "Unknown"),
                    "selection_rate": f"{item.get('selection_rate', 0):.1%}" if not pd.isna(item.get('selection_rate')) else "Insufficient Data",
                    "risk_level": item.get("risk_level", "LOW"),
                }
                for item in top_risky
            ]
        }
    
    # 6. IMPACT ASSESSMENT
    impact_assessment = []
    for col, pop_data in affected_population.items():
        total_affected = pop_data.get("total_affected_individuals", 0)
        affected_groups = pop_data.get("affected_groups", [])
        
        group_impacts = []
        for group in affected_groups:
            group_impacts.append({
                "group": group.get("group", "Unknown"),
                "disadvantaged_count": group.get("disadvantaged_count", 0),
                "total_count": group.get("total_count", 0),
            })
        
        impact_assessment.append({
            "attribute": col,
            "total_affected_individuals": total_affected if not pd.isna(total_affected) else "Cannot estimate",
            "disadvantaged_groups": group_impacts,
            "explanation": pop_data.get("explanation", "Insufficient data for impact estimation"),
        })
    
    # 7. RECOMMENDATIONS
    urgent_actions = []
    monitor_actions = []
    safe_actions = []
    
    # Group high-risk attributes for consolidated recommendations
    high_risk_attrs = []
    moderate_risk_attrs = []
    low_risk_attrs = []
    low_confidence_attrs = []
    
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        
        severity = metrics.get("severity", "LOW")
        dp_diff = metrics.get("dp_diff", 0)
        confidence = metrics.get("confidence", "MEDIUM")
        
        if severity == "HIGH":
            high_risk_attrs.append(col)
        elif severity == "MODERATE":
            moderate_risk_attrs.append(col)
        else:
            low_risk_attrs.append(col)
        
        if confidence == "LOW":
            low_confidence_attrs.append(col)
    
    # Consolidated urgent actions
    if high_risk_attrs:
        attrs_str = ", ".join(high_risk_attrs)
        urgent_actions.append({
            "action": f"Apply reweighting or resampling for: {attrs_str}",
            "reason": f"High disparity detected across these attributes. Implement sample reweighting to balance group representation during model training.",
            "timeline": "Within 1 week",
            "priority": "CRITICAL"
        })
        
        # Check for proxy features affecting these attributes
        affected_proxies = [pf for pf in proxy_features if pf.get("sensitive_column") in high_risk_attrs]
        if affected_proxies:
            proxy_names = ", ".join([pf.get("feature", "unknown") for pf in affected_proxies])
            urgent_actions.append({
                "action": f"Audit and potentially remove proxy features: {proxy_names}",
                "reason": f"These features are highly correlated with sensitive attributes and may be encoding systemic bias. Review feature importance scores and remove or transform them.",
                "timeline": "Within 2 weeks",
                "priority": "HIGH"
            })
    
    if low_confidence_attrs:
        attrs_str = ", ".join(low_confidence_attrs)
        urgent_actions.append({
            "action": f"Collect more data for: {attrs_str} to improve confidence",
            "reason": f"High or moderate disparity detected but confidence is LOW due to small sample sizes. Collect additional data to validate findings before making major changes.",
            "timeline": "Within 1 month",
            "priority": "HIGH"
        })
    
    # Consolidated monitor actions
    if moderate_risk_attrs:
        attrs_str = ", ".join(moderate_risk_attrs)
        monitor_actions.append({
            "action": f"Monitor disparity trends for: {attrs_str} over the next 30 days",
            "reason": f"Moderate disparity should be tracked. Set up automated monitoring to detect if disparity increases beyond acceptable thresholds.",
            "timeline": "Ongoing - review in 30 days",
            "priority": "MEDIUM"
        })
        
        monitor_actions.append({
            "action": f"Validate findings for: {attrs_str} with additional datasets",
            "reason": "Cross-validation helps confirm whether disparities are consistent. Test the model on diverse datasets to ensure findings are not data-specific.",
            "timeline": "Within 2 weeks",
            "priority": "MEDIUM"
        })
    
    if low_risk_attrs:
        attrs_str = ", ".join(low_risk_attrs)
        safe_actions.append({
            "action": f"Continue standard monitoring for: {attrs_str}",
            "reason": f"Low disparity indicates fair outcomes. Maintain current practices and continue regular bias audits.",
            "timeline": "Quarterly review",
            "priority": "LOW"
        })
    
    # Add data issue recommendations
    for col in sensitive_columns:
        missing_pct = dataframe[col].isna().sum() / len(dataframe) * 100
        if missing_pct > 5:
            urgent_actions.append({
                "action": f"Handle missing values in {col} using imputation",
                "reason": f"{missing_pct:.1f}% missing data may bias results. Use median/mode imputation or create a separate 'unknown' category to avoid bias from exclusion.",
                "timeline": "Within 1 week",
                "priority": "HIGH"
            })
    
    # 8. DATA ISSUES HANDLING
    data_issues = []
    for col, metrics in fairness_metrics.items():
        if col not in sensitive_columns:
            continue
        
        di_ratio = metrics.get("di_ratio", 1.0)
        group_analysis = metrics.get("group_analysis", [])
        
        # Check for extreme DI ratio
        if pd.isna(di_ratio) or di_ratio == 0 or di_ratio > 10:
            data_issues.append({
                "attribute": col,
                "issue": "Extreme Disparate Impact Ratio",
                "explanation": f"DI ratio of {di_ratio if not pd.isna(di_ratio) else 'N/A'} indicates extreme imbalance, possibly due to very small group sizes or data quality issues."
            })
        
        # Check for low sample sizes
        low_sample_groups = [g for g in group_analysis if g.get("total_samples", 0) < 30]
        if low_sample_groups:
            data_issues.append({
                "attribute": col,
                "issue": "Low Sample Size Warning",
                "explanation": f"{len(low_sample_groups)} groups have fewer than 30 samples, which may lead to unreliable conclusions."
            })
    
    # Generate executive summary
    executive_summary = ""
    if high_severity:
        top_risk = high_severity[0]
        executive_summary = f"High risk detected, primarily driven by {top_risk} disparity. Immediate action required to ensure fair outcomes."
    elif moderate_severity:
        top_risk = moderate_severity[0]
        executive_summary = f"Moderate risk detected, primarily driven by {top_risk} disparity. Monitor trends and validate findings."
    else:
        executive_summary = "No significant bias detected. System shows fair outcomes across all analyzed attributes."

    # Generate recommended decision
    recommended_decision = ""
    if high_severity:
        attrs_str = ", ".join(high_severity)
        recommended_decision = f"🚨 Proceed only after mitigation of {attrs_str} bias"
    elif moderate_severity:
        attrs_str = ", ".join(moderate_severity)
        recommended_decision = f"⚠️ Proceed with monitoring of {attrs_str} trends"
    else:
        recommended_decision = "✅ Safe to proceed with current system"

    # Add reliability warning if low confidence detected
    reliability_warning = ""
    if low_confidence_attrs:
        attrs_str = ", ".join(low_confidence_attrs)
        reliability_warning = f"⚠️ Results may be unreliable due to extremely small sample sizes in {attrs_str}. Collect additional data before making critical decisions."

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
