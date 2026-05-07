"""
Explainable AI (XAI) API routes.

Provides endpoints for:
- SHAP-based feature importance explanations
- Per-group feature importance analysis
- Counterfactual explanations
- Individual prediction explanations
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional, List
import json
import pandas as pd
from app.services.inference.data_loader import load_csv_from_upload
from app.services.xai.shap_explainer import (
    explain_predictions,
    analyze_feature_importance_by_group,
    prepare_features,
    train_surrogate_model,
)
from app.services.xai.counterfactual import (
    generate_counterfactuals,
    find_minimum_changes,
)
from app.core.logging import logger

router = APIRouter()


@router.post("/explain/feature-importance")
async def feature_importance_explanation(
    file: UploadFile = File(...),
    prediction_column: Optional[str] = Form(default=None),
    sensitive_column: Optional[str] = Form(default=None),
    feature_columns: Optional[str] = Form(default=None),
):
    """
    Get SHAP-based feature importance analysis, broken down by demographic group.
    
    This reveals if the model uses different features for different groups,
    which can indicate discriminatory patterns.
    
    Upload a CSV with:
    - Model predictions (0/1 or binary labels)
    - Sensitive attribute columns (gender, race, age_group, etc.)
    - Feature columns used by the model
    
    Returns:
    - Overall top features
    - Per-group feature importance rankings
    - Comparative analysis showing divergent feature usage
    - Potential bias indicators
    """
    try:
        df = load_csv_from_upload(file)
        logger.info(f"Feature importance request: {len(df)} rows")
        
        # Auto-detect columns if not provided
        cols = list(df.columns)
        
        pred_col = prediction_column
        if not pred_col:
            # Try common prediction column names
            for hint in ["prediction", "predicted", "pred", "score", "output", "y_pred"]:
                matches = [c for c in cols if hint in c.lower()]
                if matches:
                    pred_col = matches[0]
                    break
        
        if not pred_col or pred_col not in df.columns:
            raise ValueError(
                "Could not detect prediction column. "
                "Please specify prediction_column parameter."
            )
        
        # Parse feature columns
        if feature_columns:
            feature_cols = [c.strip() for c in feature_columns.split(",") if c.strip()]
        else:
            # Use all non-prediction numeric columns as features
            feature_cols = [
                c for c in cols
                if c != pred_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        if not feature_cols:
            raise ValueError("No feature columns found. Please specify feature_columns parameter.")
        
        # Train surrogate model for explanation
        X, encoders = prepare_features(df, feature_cols, exclude_columns=[pred_col])
        model = train_surrogate_model(X, df[pred_col])
        
        result = {}
        
        # If sensitive column provided, do per-group analysis
        if sensitive_column and sensitive_column in df.columns:
            result = analyze_feature_importance_by_group(
                df=df,
                model=model,
                sensitive_column=sensitive_column,
                feature_columns=feature_cols,
                exclude_columns=[pred_col],
            )
        else:
            # Overall analysis only
            from app.services.xai.shap_explainer import compute_shap_values
            shap_values, explainer = compute_shap_values(X, model)
            overall_importance = abs(shap_values).mean(axis=0)
            
            result = {
                "overall_top_features": [
                    {"feature": name, "importance": float(imp)}
                    for name, imp in sorted(
                        zip(X.columns, overall_importance),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                ],
                "note": "Provide sensitive_column for per-group analysis",
            }
        
        result["model_info"] = {
            "surrogate_type": "random_forest",
            "samples": len(df),
            "features": len(feature_cols),
            "prediction_column": pred_col,
        }
        
        return result
        
    except ValueError as exc:
        logger.error(f"Validation error in feature_importance: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in feature_importance: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


@router.post("/explain/predictions")
async def explain_individual_predictions(
    file: UploadFile = File(...),
    prediction_column: Optional[str] = Form(default=None),
    feature_columns: Optional[str] = Form(default=None),
    sample_indices: Optional[str] = Form(default=None),
    num_samples: int = Form(default=5),
):
    """
    Generate SHAP explanations for individual predictions.
    
    Shows which features contributed most to each prediction,
    helping identify why specific decisions were made.
    
    Args:
        sample_indices: Comma-separated list of row indices to explain
        num_samples: Number of random samples if indices not provided
    
    Returns:
    - Explanation for each sample
    - Top positive/negative contributing features
    - Human-readable explanation summaries
    """
    try:
        df = load_csv_from_upload(file)
        
        # Detect prediction column
        pred_col = prediction_column
        if not pred_col:
            for hint in ["prediction", "predicted", "pred", "score", "output", "y_pred"]:
                matches = [c for c in df.columns if hint in c.lower()]
                if matches:
                    pred_col = matches[0]
                    break
        
        if not pred_col or pred_col not in df.columns:
            raise ValueError("Could not detect prediction column. Please specify prediction_column.")
        
        # Parse feature columns
        if feature_columns:
            feature_cols = [c.strip() for c in feature_columns.split(",") if c.strip()]
        else:
            feature_cols = [
                c for c in df.columns
                if c != pred_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        # Parse sample indices
        indices = None
        if sample_indices:
            try:
                indices = [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
            except ValueError:
                raise ValueError("sample_indices must be comma-separated integers")
        
        # Train surrogate model
        X, _ = prepare_features(df, feature_cols, exclude_columns=[pred_col])
        model = train_surrogate_model(X, df[pred_col])
        
        result = explain_predictions(
            df=df,
            model=model,
            prediction_col=pred_col,
            feature_columns=feature_cols,
            sample_indices=indices,
            num_samples=num_samples,
        )
        
        return result
        
    except ValueError as exc:
        logger.error(f"Validation error in explain_predictions: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in explain_predictions: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


@router.post("/explain/counterfactuals")
async def generate_counterfactual_explanations(
    file: UploadFile = File(...),
    prediction_column: Optional[str] = Form(default=None),
    feature_columns: Optional[str] = Form(default=None),
    sensitive_columns: Optional[str] = Form(default=""),
    instance_index: int = Form(...),
    desired_outcome: Optional[int] = Form(default=None),
    num_counterfactuals: int = Form(default=5),
):
    """
    Generate counterfactual explanations for a specific instance.
    
    Counterfactuals answer: "What would need to change for this outcome to be different?"
    
    When sensitive attributes appear in these changes, it may indicate bias.
    
    Example response:
    ```
    {
      "original_instance": {...},
      "current_prediction": 0,
      "desired_outcome": 1,
      "counterfactuals": [
        {
          "changes": {
            "age": {"from": 25, "to": 35, "change_type": "increase"},
            "income": {"from": 30000, "to": 45000, "change_type": "increase"}
          },
          "num_features_changed": 2
        }
      ],
      "bias_analysis": {
        "would_need_to_change_sensitive": false,
        "fairness_concerns": []
      },
      "summary": "To achieve approval, age would need to increase from 25 to 35..."
    }
    ```
    
    Returns:
    - Counterfactual examples from the dataset
    - Bias analysis (if sensitive attributes would need to change)
    - Human-readable summary
    """
    try:
        df = load_csv_from_upload(file)
        
        if instance_index < 0 or instance_index >= len(df):
            raise ValueError(f"instance_index {instance_index} out of range (0-{len(df)-1})")
        
        # Detect prediction column
        pred_col = prediction_column
        if not pred_col:
            for hint in ["prediction", "predicted", "pred", "score", "output", "y_pred"]:
                matches = [c for c in df.columns if hint in c.lower()]
                if matches:
                    pred_col = matches[0]
                    break
        
        if not pred_col or pred_col not in df.columns:
            raise ValueError("Could not detect prediction column. Please specify prediction_column.")
        
        # Parse feature columns
        if feature_columns:
            feature_cols = [c.strip() for c in feature_columns.split(",") if c.strip()]
        else:
            feature_cols = [
                c for c in df.columns
                if c != pred_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        # Parse sensitive columns
        sens_cols = []
        if sensitive_columns:
            sens_cols = [c.strip() for c in sensitive_columns.split(",") if c.strip()]
        
        # Train surrogate model
        from app.services.xai.shap_explainer import prepare_features
        X, _ = prepare_features(df, feature_cols, exclude_columns=[pred_col])
        model = train_surrogate_model(X, df[pred_col])
        
        # Add predictions to dataframe for counterfactual search
        df_copy = df.copy()
        df_copy["prediction"] = model.predict(X)
        
        instance = df_copy.iloc[instance_index]
        
        result = generate_counterfactuals(
            df=df_copy,
            model=model,
            instance=instance,
            desired_outcome=desired_outcome,
            sensitive_columns=sens_cols,
            immutable_features=sens_cols,  # Sensitive attrs shouldn't need to change
            num_counterfactuals=num_counterfactuals,
        )
        
        # Add instance metadata
        result["instance_index"] = instance_index
        result["total_dataset_size"] = len(df)
        
        return result
        
    except ValueError as exc:
        logger.error(f"Validation error in counterfactuals: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in counterfactuals: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


@router.post("/explain/minimal-changes")
async def minimal_change_explanation(
    file: UploadFile = File(...),
    prediction_column: Optional[str] = Form(default=None),
    feature_columns: Optional[str] = Form(default=None),
    instance_index: int = Form(...),
    desired_outcome: int = Form(...),
):
    """
    Find the minimum changes needed to flip a prediction.
    
    Uses optimization to find the smallest set of feature changes
    that would result in a different outcome.
    
    This helps understand which features the model is most sensitive to
    for a specific decision.
    
    Returns:
    - Minimal feature changes
    - Magnitude of required changes
    - Success/failure status
    """
    try:
        df = load_csv_from_upload(file)
        
        if instance_index < 0 or instance_index >= len(df):
            raise ValueError(f"instance_index {instance_index} out of range (0-{len(df)-1})")
        
        # Detect prediction column
        pred_col = prediction_column
        if not pred_col:
            for hint in ["prediction", "predicted", "pred", "score", "output", "y_pred"]:
                matches = [c for c in df.columns if hint in c.lower()]
                if matches:
                    pred_col = matches[0]
                    break
        
        if not pred_col or pred_col not in df.columns:
            raise ValueError("Could not detect prediction column.")
        
        # Parse feature columns
        if feature_columns:
            feature_cols = [c.strip() for c in feature_columns.split(",") if c.strip()]
        else:
            feature_cols = [
                c for c in df.columns
                if c != pred_col and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        # Train surrogate model
        from app.services.xai.shap_explainer import prepare_features
        X, _ = prepare_features(df, feature_cols, exclude_columns=[pred_col])
        model = train_surrogate_model(X, df[pred_col])
        
        instance = df.iloc[instance_index]
        
        # Compute feature ranges from data
        feature_ranges = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_ranges[col] = (float(df[col].min()), float(df[col].max()))
        
        result = find_minimum_changes(
            df=df,
            model=model,
            instance=instance,
            desired_outcome=desired_outcome,
            feature_ranges=feature_ranges,
        )
        
        result["instance_index"] = instance_index
        result["instance_preview"] = {k: v for k, v in instance.items() if k in feature_cols[:5]}
        
        return result
        
    except ValueError as exc:
        logger.error(f"Validation error in minimal_changes: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error in minimal_changes: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


@router.get("/explain/info")
async def explain_info():
    """Get information about available XAI endpoints and their usage."""
    return {
        "endpoints": [
            {
                "path": "/explain/feature-importance",
                "method": "POST",
                "description": "SHAP-based feature importance by demographic group",
                "file_upload": True,
                "parameters": ["prediction_column", "sensitive_column", "feature_columns"],
            },
            {
                "path": "/explain/predictions",
                "method": "POST",
                "description": "Individual prediction explanations with SHAP",
                "file_upload": True,
                "parameters": ["prediction_column", "feature_columns", "sample_indices", "num_samples"],
            },
            {
                "path": "/explain/counterfactuals",
                "method": "POST",
                "description": "Counterfactual explanations - what would need to change",
                "file_upload": True,
                "parameters": ["prediction_column", "instance_index", "desired_outcome", "sensitive_columns"],
            },
            {
                "path": "/explain/minimal-changes",
                "method": "POST",
                "description": "Minimum feature changes to flip prediction",
                "file_upload": True,
                "parameters": ["prediction_column", "instance_index", "desired_outcome"],
            },
        ],
        "supported_file_formats": ["CSV"],
        "note": "All endpoints train a surrogate Random Forest model when the original model is not available.",
    }
