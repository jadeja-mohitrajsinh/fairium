import pandas as pd
import numpy as np
from app.core.logging import logger

try:
    import shap
    import xgboost as xgb
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP or XGBoost not installed. XAI features will be disabled.")

def generate_shap_importance(dataframe: pd.DataFrame, target_column: str, max_features: int = 10) -> list:
    """Train a quick model and generate SHAP feature importance."""
    if not SHAP_AVAILABLE:
        return []

    try:
        df_clean = dataframe.copy()
        
        # Very basic preprocessing for the quick model
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' or str(df_clean[col].dtype) == 'category':
                df_clean[col] = df_clean[col].astype('category').cat.codes
                
        df_clean = df_clean.dropna(subset=[target_column])
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
        X = X.fillna(X.median())
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        X_sample = shap.sample(X, 100) if len(X) > 100 else X
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        feature_importance = []
        for i, feature in enumerate(X.columns):
            feature_importance.append({
                "feature": feature,
                "importance": float(mean_shap[i])
            })
            
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance[:max_features]
    except Exception as e:
        logger.error(f"Error calculating SHAP: {e}")
        return []

def calculate_accuracy_fairness_tradeoff(dataframe: pd.DataFrame, target_column: str, sensitive_column: str) -> list:
    """Simulate different mitigation thresholds to show tradeoff curve."""
    # Simplified simulation for the UI slider
    base_accuracy = 0.85
    base_fairness = 0.60
    
    tradeoff_curve = []
    tradeoff_curve.append({"mitigation_level": 0, "accuracy": base_accuracy, "fairness": base_fairness})
    tradeoff_curve.append({"mitigation_level": 25, "accuracy": base_accuracy - 0.01, "fairness": base_fairness + 0.10})
    tradeoff_curve.append({"mitigation_level": 50, "accuracy": base_accuracy - 0.03, "fairness": base_fairness + 0.20})
    tradeoff_curve.append({"mitigation_level": 75, "accuracy": base_accuracy - 0.06, "fairness": base_fairness + 0.30})
    tradeoff_curve.append({"mitigation_level": 100, "accuracy": base_accuracy - 0.12, "fairness": 1.0})
    
    return tradeoff_curve
