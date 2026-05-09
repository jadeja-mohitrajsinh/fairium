"""Counterfactual explanation generation for bias analysis.

Counterfactual explanations answer "what would need to change for this outcome
to be different?" This is particularly useful for bias detection as it can reveal
when protected attributes (or their proxies) need to change for fair outcomes.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from app.core.logging import logger


def generate_counterfactuals(
    df: pd.DataFrame,
    model: Any,
    instance: pd.Series,
    desired_outcome: Optional[int] = None,
    sensitive_columns: Optional[List[str]] = None,
    immutable_features: Optional[List[str]] = None,
    max_changes: int = 3,
    num_counterfactuals: int = 5,
) -> Dict[str, Any]:
    """
    Generate counterfactual explanations for a single instance.
    
    Finds minimal changes to flip the prediction outcome. When sensitive
    attributes appear in these changes, it may indicate discriminatory patterns.
    
    Args:
        df: Reference dataset
        model: Trained model with predict method
        instance: Single row (pd.Series) to explain
        desired_outcome: Target prediction (0 or 1). If None, flips current prediction.
        sensitive_columns: List of protected attributes to monitor
        immutable_features: Features that cannot change (e.g., race, gender)
        max_changes: Maximum number of features to change
        num_counterfactuals: Number of counterfactuals to generate
    
    Returns:
        Dict with counterfactual explanations and bias analysis
    """
    sensitive_columns = sensitive_columns or []
    immutable_features = immutable_features or sensitive_columns.copy()
    
    # Get current prediction
    current_pred = _get_prediction(model, instance)
    
    if desired_outcome is None:
        desired_outcome = 1 - current_pred
    
    # Find counterfactuals
    counterfactuals = _search_counterfactuals(
        df=df,
        model=model,
        instance=instance,
        desired_outcome=desired_outcome,
        immutable_features=immutable_features,
        max_changes=max_changes,
        num_counterfactuals=num_counterfactuals,
    )
    
    # Analyze for bias indicators
    bias_analysis = _analyze_counterfactual_bias(
        instance=instance,
        counterfactuals=counterfactuals,
        sensitive_columns=sensitive_columns,
        immutable_features=immutable_features,
    )
    
    return {
        "original_instance": instance.to_dict(),
        "current_prediction": int(current_pred),
        "desired_outcome": int(desired_outcome),
        "counterfactuals": counterfactuals,
        "bias_analysis": bias_analysis,
        "summary": _generate_counterfactual_summary(counterfactuals, bias_analysis),
    }


def find_minimum_changes(
    df: pd.DataFrame,
    model: Any,
    instance: pd.Series,
    desired_outcome: int,
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Find the minimum set of feature changes to flip the prediction.
    
    Uses a greedy approach to find minimal perturbations. This reveals
    which features the model is most sensitive to.
    
    Args:
        df: Reference dataset
        model: Trained model
        instance: Instance to modify
        desired_outcome: Target prediction
        feature_ranges: Valid range for each feature (min, max)
        max_iterations: Maximum optimization iterations
    
    Returns:
        Dict with minimal changes and analysis
    """
    feature_cols = [c for c in df.columns if c != 'prediction' and c != 'target']
    original_pred = _get_prediction(model, instance)
    
    if original_pred == desired_outcome:
        return {
            "message": "Instance already has the desired outcome",
            "changes_needed": [],
        }
    
    # Determine feature ranges from data if not provided
    if feature_ranges is None:
        feature_ranges = {
            col: (df[col].min(), df[col].max())
            for col in feature_cols
            if pd.api.types.is_numeric_dtype(df[col])
        }
    
    best_changes = {}
    best_score = float('inf')
    
    # Greedy search: try changing each feature individually
    for col in feature_cols:
        if col not in feature_ranges:
            continue
            
        min_val, max_val = feature_ranges[col]
        current_val = instance[col]
        
        # Try values across the range
        test_values = np.linspace(min_val, max_val, 20)
        
        for test_val in test_values:
            if abs(test_val - current_val) < 0.001:
                continue
                
            modified = instance.copy()
            modified[col] = test_val
            
            # Check if prediction flips
            new_pred = _get_prediction(model, modified)
            
            if new_pred == desired_outcome:
                change_magnitude = abs(test_val - current_val)
                if change_magnitude < best_score:
                    best_score = change_magnitude
                    best_changes = {
                        col: {
                            "original": float(current_val),
                            "new": float(test_val),
                            "change": float(test_val - current_val),
                        }
                    }
    
    # Try pairs of features if single changes don't work
    if not best_changes:
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                if col1 not in feature_ranges or col2 not in feature_ranges:
                    continue
                    
                result = _try_two_feature_change(
                    model, instance, col1, col2,
                    feature_ranges[col1], feature_ranges[col2],
                    desired_outcome
                )
                
                if result and result["score"] < best_score:
                    best_score = result["score"]
                    best_changes = result["changes"]
    
    return {
        "original_prediction": int(original_pred),
        "desired_outcome": int(desired_outcome),
        "minimal_changes": best_changes,
        "total_features_changed": len(best_changes),
        "success": len(best_changes) > 0,
        "explanation": _generate_minimal_change_explanation(best_changes),
    }


def _search_counterfactuals(
    df: pd.DataFrame,
    model: Any,
    instance: pd.Series,
    desired_outcome: int,
    immutable_features: List[str],
    max_changes: int,
    num_counterfactuals: int,
) -> List[Dict[str, Any]]:
    """Search for counterfactual instances in the dataset."""
    
    # Filter to only instances with desired outcome
    feature_cols = [c for c in df.columns if c not in ['prediction', 'target', 'actual']]
    
    # Get predictions for all rows if not present
    if 'prediction' not in df.columns:
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        df = df.copy()
        df['prediction'] = model.predict(X)
    
    desired_df = df[df['prediction'] == desired_outcome].copy()
    
    if len(desired_df) == 0:
        logger.warning("No instances with desired outcome found in dataset")
        return []
    
    # Find nearest neighbors with desired outcome
    X_desired = desired_df[feature_cols].fillna(desired_df[feature_cols].median(numeric_only=True))
    
    # Encode categorical variables
    for col in X_desired.columns:
        if X_desired[col].dtype == 'object':
            X_desired[col] = pd.Categorical(X_desired[col]).codes
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(num_counterfactuals * 3, len(X_desired)), metric='euclidean')
    nn.fit(X_desired)
    
    # Prepare instance
    instance_features = instance[feature_cols].copy()
    instance_features = instance_features.fillna(0)
    for col in instance_features.index:
        if col in X_desired.columns:
            if pd.api.types.is_numeric_dtype(X_desired[col]):
                try:
                    instance_features[col] = float(instance_features[col])
                except (ValueError, TypeError):
                    instance_features[col] = 0
    
    # Find neighbors
    distances, indices = nn.kneighbors([instance_features.values])
    
    counterfactuals = []
    for dist, idx in zip(distances[0], indices[0]):
        cf_instance = desired_df.iloc[idx]
        
        # Calculate changes
        changes = {}
        for col in feature_cols:
            if col in immutable_features:
                continue
                
            orig_val = instance[col]
            cf_val = cf_instance[col]
            
            if pd.isna(orig_val) or pd.isna(cf_val):
                continue
                
            if orig_val != cf_val:
                changes[col] = {
                    "from": _serialize_value(orig_val),
                    "to": _serialize_value(cf_val),
                    "change_type": _classify_change(orig_val, cf_val),
                }
        
        if len(changes) <= max_changes:
            counterfactuals.append({
                "distance": float(dist),
                "changes": changes,
                "num_features_changed": len(changes),
                "changed_features": list(changes.keys()),
                "instance": cf_instance.to_dict(),
            })
        
        if len(counterfactuals) >= num_counterfactuals:
            break
    
    # Sort by distance and number of changes
    counterfactuals.sort(key=lambda x: (x["num_features_changed"], x["distance"]))
    
    return counterfactuals[:num_counterfactuals]


def _analyze_counterfactual_bias(
    instance: pd.Series,
    counterfactuals: List[Dict],
    sensitive_columns: List[str],
    immutable_features: List[str],
) -> Dict[str, Any]:
    """Analyze counterfactuals for bias indicators."""
    
    analysis = {
        "sensitive_attributes_in_changes": [],
        "would_need_to_change_sensitive": False,
        "proxies_for_sensitive": [],
        "fairness_concerns": [],
    }
    
    # Check if sensitive attributes appear in changes
    for cf in counterfactuals:
        for feature in cf.get("changed_features", []):
            if feature in sensitive_columns:
                analysis["sensitive_attributes_in_changes"].append({
                    "feature": feature,
                    "original_value": _serialize_value(instance[feature]),
                    "change": cf["changes"][feature],
                })
                analysis["would_need_to_change_sensitive"] = True
    
    # Detect potential proxies (features that change together with sensitive attrs)
    if analysis["would_need_to_change_sensitive"]:
        analysis["fairness_concerns"].append({
            "type": "direct_sensitive_change",
            "severity": "HIGH",
            "explanation": (
                f"To achieve the desired outcome, the {analysis['sensitive_attributes_in_changes'][0]['feature']} "
                f"would need to change from '{analysis['sensitive_attributes_in_changes'][0]['original_value']}' "
                f"to '{analysis['sensitive_attributes_in_changes'][0]['change']['to']}'. "
                f"This suggests direct bias based on a protected attribute."
            ),
        })
    
    # Check for correlated features (potential proxies)
    all_changed_features = set()
    for cf in counterfactuals:
        all_changed_features.update(cf.get("changed_features", []))
    
    non_sensitive_changes = all_changed_features - set(sensitive_columns)
    if len(non_sensitive_changes) > 0 and analysis["would_need_to_change_sensitive"]:
        analysis["proxies_for_sensitive"] = list(non_sensitive_changes)
        analysis["fairness_concerns"].append({
            "type": "proxy_discrimination",
            "severity": "MODERATE",
            "explanation": (
                f"Changes to non-sensitive features {list(non_sensitive_changes)} are required "
                f"along with sensitive attribute changes. These may be acting as proxies."
            ),
        })
    
    return analysis


def _try_two_feature_change(
    model: Any,
    instance: pd.Series,
    col1: str,
    col2: str,
    range1: Tuple[float, float],
    range2: Tuple[float, float],
    desired_outcome: int,
) -> Optional[Dict]:
    """Try changing two features simultaneously."""
    
    best_result = None
    best_score = float('inf')
    
    orig1, orig2 = instance[col1], instance[col2]
    
    vals1 = np.linspace(range1[0], range1[1], 10)
    vals2 = np.linspace(range2[0], range2[1], 10)
    
    for v1 in vals1:
        for v2 in vals2:
            if abs(v1 - orig1) < 0.001 and abs(v2 - orig2) < 0.001:
                continue
                
            modified = instance.copy()
            modified[col1] = v1
            modified[col2] = v2
            
            new_pred = _get_prediction(model, modified)
            
            if new_pred == desired_outcome:
                score = abs(v1 - orig1) + abs(v2 - orig2)
                if score < best_score:
                    best_score = score
                    best_result = {
                        "score": score,
                        "changes": {
                            col1: {"original": float(orig1), "new": float(v1), "change": float(v1 - orig1)},
                            col2: {"original": float(orig2), "new": float(v2), "change": float(v2 - orig2)},
                        },
                    }
    
    return best_result


def _get_prediction(model: Any, instance: pd.Series) -> int:
    """Get prediction from model for a single instance."""
    X = pd.DataFrame([instance])
    
    # Handle categorical variables
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    X = X.fillna(0)
    
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        X = X[[c for c in expected_features if c in X.columns]]
    else:
        for col in ['prediction', 'target', 'actual']:
            if col in X.columns:
                X = X.drop(columns=[col])
    
    pred = model.predict(X)
    
    if hasattr(pred, '__iter__'):
        pred = pred[0]
    
    return int(round(float(pred)))


def _serialize_value(val: Any) -> Any:
    """Serialize a value for JSON output."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.floating)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _classify_change(from_val: Any, to_val: Any) -> str:
    """Classify the type of change."""
    try:
        f, t = float(from_val), float(to_val)
        if t > f:
            return "increase"
        elif t < f:
            return "decrease"
        else:
            return "no_change"
    except (ValueError, TypeError):
        return "changed" if from_val != to_val else "no_change"


def _generate_counterfactual_summary(
    counterfactuals: List[Dict],
    bias_analysis: Dict,
) -> str:
    """Generate human-readable summary of counterfactual analysis."""
    
    if not counterfactuals:
        return "No counterfactual examples found in the dataset."
    
    cf = counterfactuals[0]
    num_changes = cf["num_features_changed"]
    changed = cf["changed_features"]
    
    parts = [f"To achieve a different outcome, approximately {num_changes} features would need to change:"]
    
    for feature in changed[:3]:
        change = cf["changes"][feature]
        parts.append(f"  - {feature}: from {change['from']} to {change['to']}")
    
    # Add bias concerns
    if bias_analysis["would_need_to_change_sensitive"]:
        sensitive = bias_analysis["sensitive_attributes_in_changes"][0]
        parts.append(
            f"\n⚠️ BIAS ALERT: The '{sensitive['feature']}' attribute would need to change "
            f"(from '{sensitive['original_value']}' to '{sensitive['change']['to']}'). "
            f"This indicates the decision may depend on a protected characteristic."
        )
    
    return "\n".join(parts)


def _generate_minimal_change_explanation(changes: Dict) -> str:
    """Generate explanation for minimal changes."""
    if not changes:
        return "No minimal changes found to flip the prediction."
    
    parts = ["To change the prediction, you would need to:"]
    
    for feature, details in changes.items():
        direction = "increase" if details["change"] > 0 else "decrease"
        parts.append(
            f"  - {direction} {feature} from {details['original']:.2f} to {details['new']:.2f}"
        )
    
    return "\n".join(parts)
