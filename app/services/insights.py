"""Enhanced bias insights generation for human-readable interpretation."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


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
        explanation += f"This represents a {ratio:.1f}× difference in outcomes."
    elif ratio < 5:
        explanation += f"This represents a {ratio:.1f}× difference in outcomes, indicating a notable disparity."
    else:
        explanation += f"This represents a {ratio:.1f}× difference in outcomes, indicating a significant disparity."
    
    # Add sample size context
    highest_samples = next((g["total_samples"] for g in group_analysis if g["group"] == highest_group), 0)
    lowest_samples = next((g["total_samples"] for g in group_analysis if g["group"] == lowest_group), 0)
    
    if highest_samples > 0 and lowest_samples > 0:
        explanation += (f" Results are based on {highest_samples} samples for '{highest_group}' "
                      f"and {lowest_samples} for '{lowest_group}'.")
    
    return explanation
