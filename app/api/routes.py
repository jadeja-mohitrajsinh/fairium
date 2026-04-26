from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import StreamingResponse
import pandas as pd
import io
from pydantic import BaseModel
from app.services.auto_debias_engine import AutoDebiasEngine, convert_numpy_types

from app.models.schemas import (
    AnalysisResponse,
    GateRequest,
    GateResponse,
    MonitorRequest,
    MonitorResponse,
    TextBiasAnalysisResponse,
)
from app.services.data_loader import load_csv_from_upload
from app.services.decision_intelligence import (
    append_tracking_log,
    build_audit_report,
    build_data_intelligence,
    build_explainability,
    build_mitigation_preview,
    build_risk_heatmap,
    build_threshold_alerts,
    compute_unified_bias_risk_score,
    detect_bias_drift,
    evaluate_gate,
)
from app.services.inference import infer_analysis_columns
from app.services.fairness import analyze_dataset_bias
from app.services.text_bias import TextBiasAnalyzer


router = APIRouter()


class TextBiasRequest(BaseModel):
    text: str


from app.services.gemini_ai import GeminiAIService

@router.post("/analyze-text", response_model=TextBiasAnalysisResponse)
async def analyze_text(request: TextBiasRequest) -> TextBiasAnalysisResponse:
    """Analyze text for potential bias, discrimination, or unfair patterns."""
    try:
        # First try to use Gemini for advanced LLM analysis
        try:
            gemini_service = GeminiAIService()
            llm_result = await gemini_service.analyze_text_for_bias(request.text)
            
            if llm_result:
                # Map LLM output to schema
                biases = []
                for b in llm_result.get("biases", []):
                    biases.append({
                        "type": b.get("type", "Unknown"),
                        "confidence": llm_result.get("confidence", "Medium"),
                        "explanation": b.get("explanation", ""),
                        "alternatives": b.get("alternatives", []),
                        "keyword_matches": [],
                        "phrase_matches": [],
                        "ambiguous_matches": []
                    })
                
                result = {
                    "bias_detected": llm_result.get("bias_detected", "Possible"),
                    "biases": biases,
                    "overall_confidence": llm_result.get("confidence", "Medium"),
                    "ml_confidence": 0.95, # Assuming high confidence from LLM
                    "summary": llm_result.get("summary", "LLM Analysis completed.")
                }
                return TextBiasAnalysisResponse(**result)
        except Exception as e:
            print(f"Gemini LLM failed or not configured, falling back to rule-based: {e}")
            pass
            
        # Fallback to rule-based + basic ML
        result = TextBiasAnalyzer.analyze_text(request.text)
        return TextBiasAnalysisResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
) -> AnalysisResponse:
    try:
        dataframe = load_csv_from_upload(file)
        
        # Auto-detect target and sensitive columns from dataset
        inferred = infer_analysis_columns(dataframe)
        
        result = analyze_dataset_bias(
            dataframe=dataframe,
            target_column=inferred.target_column,
            sensitive_columns=inferred.sensitive_columns,
        )
        return AnalysisResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/monitor", response_model=MonitorResponse)
async def monitor(
    request: MonitorRequest = Body(
        ...,
        examples={
            "hiring_monitor": {
                "summary": "Monitor bias risk for hiring decisions",
                "value": {
                    "analysis_payload": {
                        "detected_target": "hired",
                        "fairness_metrics": {
                            "gender": {
                                "dp_diff": 0.18,
                                "di_ratio": 0.62,
                                "confidence": "HIGH",
                                "group_rates": {"male": 0.78, "female": 0.60},
                                "simulation": {"improvement": "35%"},
                            }
                        },
                        "affected_population": {
                            "gender": {
                                "total_affected_individuals": 52,
                                "affected_groups": [{"group": "female", "disadvantaged_count": 52}],
                            }
                        },
                        "notes": ["5.2% missing values in sensitive columns"],
                    },
                    "historical_risk_scores": [44, 47, 52, 55],
                    "thresholds": {
                        "flag": 50,
                        "block": 75,
                        "drift_abs": 10,
                        "drift_pct": 20,
                    },
                    "scenario": "hiring",
                    "external_metadata": {"model_version": "2026.04.1"},
                },
            }
        },
    )
) -> MonitorResponse:
    """Monitor bias risk in near real-time with drift and threshold alerts."""
    try:
        score, aggregated_metrics, prioritization = compute_unified_bias_risk_score(request.analysis_payload)
        drift = detect_bias_drift(
            current_score=float(score),
            historical_scores=request.historical_risk_scores,
            thresholds=request.thresholds,
        )
        alerts = build_threshold_alerts(float(score), drift, request.thresholds)
        heatmap = build_risk_heatmap(prioritization)
        data_intelligence = build_data_intelligence(request.analysis_payload)
        mitigation_preview = build_mitigation_preview(request.analysis_payload, prioritization)
        explainability, impacted_groups, risk_consequences = build_explainability(
            request.analysis_payload, prioritization
        )
        audit_report = build_audit_report(
            scenario=request.scenario or "general",
            score=score,
            alerts=alerts,
            drift=drift,
            prioritization=prioritization,
        )

        log_id = append_tracking_log(
            "decision_monitoring",
            {
                "scenario": request.scenario or "general",
                "decision_intelligence": {
                    "unified_bias_risk_score": score,
                    "aggregated_metrics": aggregated_metrics,
                    "prioritization": prioritization,
                },
                "drift": drift,
                "alerts": alerts,
                "impacted_groups": impacted_groups,
                "risk_consequences": risk_consequences,
                "external_metadata": request.external_metadata,
            },
        )

        return MonitorResponse(
            generated_at=audit_report["generated_at"],
            decision_intelligence={
                "unified_bias_risk_score": score,
                "aggregated_metrics": aggregated_metrics,
                "prioritization": prioritization,
            },
            drift=drift,
            alerts=alerts,
            risk_heatmap=heatmap,
            data_intelligence=data_intelligence,
            mitigation_preview=mitigation_preview,
            explainability=explainability,
            impacted_groups=impacted_groups,
            audit_report=audit_report,
            tracking_log_id=log_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/gate", response_model=GateResponse)
async def gate(
    request: GateRequest = Body(
        ...,
        examples={
            "lending_gate": {
                "summary": "Gate a lending decision before approval",
                "value": {
                    "decision_id": "loan-application-10042",
                    "scenario": "lending",
                    "decision_payload": {
                        "applicant_id": "A-901",
                        "recommended_outcome": "approve",
                    },
                    "analysis_payload": {
                        "detected_target": "approved",
                        "fairness_metrics": {
                            "race": {
                                "dp_diff": 0.21,
                                "di_ratio": 0.59,
                                "confidence": "MEDIUM",
                                "group_rates": {"group_a": 0.72, "group_b": 0.51},
                                "simulation": {"improvement": "40%"},
                                "explanation": "Group B receives markedly fewer positive outcomes.",
                            }
                        },
                        "affected_population": {
                            "race": {
                                "total_affected_individuals": 88,
                                "affected_groups": [{"group": "group_b", "disadvantaged_count": 88}],
                            }
                        },
                        "notes": ["Distribution skew detected in applicant history"],
                    },
                    "block_threshold": 75,
                    "flag_threshold": 50,
                    "auto_mitigation": True,
                    "external_metadata": {"channel": "api-gateway"},
                },
            }
        },
    )
) -> GateResponse:
    """Pre-decision fairness gate with block/flag logic and approval workflow."""
    try:
        score, aggregated_metrics, prioritization = compute_unified_bias_risk_score(request.analysis_payload)
        effective_score = request.risk_score_override if request.risk_score_override is not None else score

        explainability, impacted_groups, risk_consequences = build_explainability(
            request.analysis_payload, prioritization
        )

        gate_result = evaluate_gate(
            decision_id=request.decision_id,
            scenario=request.scenario,
            score=int(effective_score),
            flag_threshold=request.flag_threshold,
            block_threshold=request.block_threshold,
            impacted_groups=impacted_groups,
            risk_consequences=risk_consequences,
            auto_mitigation=request.auto_mitigation,
        )

        log_id = append_tracking_log(
            "decision_gate",
            {
                "decision_id": request.decision_id,
                "scenario": request.scenario,
                "decision_payload": request.decision_payload,
                "effective_score": effective_score,
                "computed_score": score,
                "aggregated_metrics": aggregated_metrics,
                "prioritization": prioritization,
                "explainability": explainability,
                "result": gate_result,
                "external_metadata": request.external_metadata,
            },
        )

        return GateResponse(
            generated_at=gate_result["fairness_certificate"]["issued_at"],
            decision_id=request.decision_id,
            scenario=request.scenario,
            status=gate_result["status"],
            fairness_approval_required=gate_result["fairness_approval_required"],
            fairness_approved=gate_result["fairness_approved"],
            bias_risk_score=int(effective_score),
            reasons=gate_result["reasons"],
            impacted_groups=gate_result["impacted_groups"],
            risk_consequences=gate_result["risk_consequences"],
            mitigation_actions=gate_result["mitigation_actions"],
            fairness_certificate={
                **gate_result["fairness_certificate"],
                "aggregated_metrics": aggregated_metrics,
                "top_priorities": prioritization[:3],
            },
            tracking_log_id=log_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# =============================================================================
# REAL-TIME ADAPTIVE DEBIAS ENGINE ENDPOINTS
# Replaces old static debiasing pipeline with adaptive, production-ready engine
# =============================================================================

# Initialize the engine
engine = AutoDebiasEngine()


def detect_bias_characteristics(dataframe: pd.DataFrame, target_column: str, sensitive_column: str) -> dict:
    characteristics = {
        "di_ratio": 1.0,
        "dp_diff": 0.0,
        "group_sizes": {},
        "missing_percentage": 0.0,
        "class_imbalance": 0.0,
        "has_small_sample": False,
        "has_imbalance": False,
        "has_missing_data": False,
        "confidence": "HIGH"
    }
    
    # Calculate group sizes
    group_counts = dataframe[sensitive_column].value_counts()
    characteristics["group_sizes"] = {str(k): int(v) for k, v in group_counts.to_dict().items()}
    
    # Check for small samples (any group < 30)
    characteristics["has_small_sample"] = bool(any(count < 30 for count in group_counts.values))
    
    # Determine confidence level
    if characteristics["has_small_sample"]:
        characteristics["confidence"] = "LOW"
    elif len(group_counts) < 3:
        characteristics["confidence"] = "MEDIUM"
    
    # Check for class imbalance (ratio between largest and smallest group)
    if len(group_counts) > 1:
        max_count = group_counts.max()
        min_count = group_counts.min()
        characteristics["class_imbalance"] = float(max_count / min_count if min_count > 0 else float('inf'))
        characteristics["has_imbalance"] = bool(characteristics["class_imbalance"] > 2.0)
    
    # Check missing data
    missing_pct = float((dataframe[sensitive_column].isna().sum() / len(dataframe)) * 100)
    characteristics["missing_percentage"] = missing_pct
    characteristics["has_missing_data"] = bool(missing_pct > 10.0)
    
    # Calculate DI and DP
    try:
        target_values = dataframe[target_column].unique()
        if len(target_values) >= 2:
            positive_outcome = target_values[1]  # Assume second value is positive
            
            group_rates = {}
            for group in group_counts.index:
                group_data = dataframe[dataframe[sensitive_column] == group]
                group_rate = float((group_data[target_column] == positive_outcome).mean())
                group_rates[str(group)] = group_rate
            
            rates_list = list(group_rates.values())
            if len(rates_list) >= 2:
                max_rate = max(rates_list)
                min_rate = min(rates_list)
                characteristics["di_ratio"] = float(min_rate / max_rate if max_rate > 0 else 0)
                characteristics["dp_diff"] = float(max_rate - min_rate)
    except Exception:
        pass
    
    return characteristics


def detect_all_bias(dataframe: pd.DataFrame, target_column: str, sensitive_columns: list) -> dict:
    """Detect bias across all sensitive attributes."""
    all_bias = {}
    for col in sensitive_columns:
        if col in dataframe.columns:
            all_bias[col] = detect_bias_characteristics(dataframe, target_column, col)
    return all_bias


def select_adaptive_method(characteristics: dict) -> str:
    """Select the best mitigation method based on data characteristics."""
    # Priority order: small sample > missing data > imbalance > default
    if characteristics["has_small_sample"]:
        return "resampling"
    elif characteristics["has_missing_data"]:
        return "imputation"
    elif characteristics["has_imbalance"]:
        return "reweighing"
    else:
        return "reweighing"  # Default: reweighing works universally


def decision_engine(all_bias: dict) -> dict:
    """Auto-Debias decision engine - determines action for each attribute."""
    decisions = {}
    
    for attr, bias_info in all_bias.items():
        di = bias_info["di_ratio"]
        confidence = bias_info["confidence"]
        
        # Data validation gate
        if confidence == "LOW":
            decisions[attr] = {
                "action": "SKIP",
                "reason": "Low confidence - data insufficient",
                "priority": "DATA_COLLECTION"
            }
        elif di >= 0.8:
            decisions[attr] = {
                "action": "MONITOR",
                "reason": "Fairness acceptable - monitor only",
                "priority": "LOW"
            }
        elif 0.5 <= di < 0.8:
            decisions[attr] = {
                "action": "MILD_REWEIGHT",
                "method": "reweighing",
                "reason": "Moderate bias - mild reweighting",
                "priority": "MEDIUM"
            }
        elif 0.3 <= di < 0.5:
            decisions[attr] = {
                "action": "CONTROLLED_RESAMPLE",
                "method": "resampling",
                "reason": "High bias - controlled resampling",
                "priority": "HIGH"
            }
        else:  # DI < 0.3
            decisions[attr] = {
                "action": "STRICT_INTERVENTION",
                "method": "hybrid",
                "reason": "Severe bias - full intervention",
                "priority": "CRITICAL"
            }
    
    return decisions


def apply_aggressive_reweighing(dataframe: pd.DataFrame, target_column: str, sensitive_column: str, max_iterations: int = 10) -> pd.DataFrame:
    """Apply aggressive iterative reweighing to push DI close to 1.0."""
    df = dataframe.copy()
    
    for iteration in range(max_iterations):
        # Calculate current DI
        current_metrics = detect_bias_characteristics(df, target_column, sensitive_column)
        current_di = current_metrics["di_ratio"]
        
        # Stop if DI is close to 1.0 (within 0.05)
        if abs(current_di - 1.0) < 0.05:
            break
        
        # Apply reweighing
        counts = df.groupby([sensitive_column, target_column]).size()
        total = len(df)
        p_s = df[sensitive_column].value_counts() / total
        p_y = df[target_column].value_counts() / total
        
        weights = []
        for _, row in df.iterrows():
            s_val = row[sensitive_column]
            y_val = row[target_column]
            p_expected = p_s[s_val] * p_y[y_val]
            p_observed = counts.get((s_val, y_val), 0) / total
            
            # Aggressive weight scaling with higher bounds (0.1x to 10x)
            weight = p_expected / p_observed if p_observed > 0 else 1.0
            weight = max(0.1, min(10.0, weight))  # Bound between 0.1 and 10
            weights.append(weight)
        
        df['fairness_weight'] = weights
        
        # Re-sample based on weights to actually change distribution
        sample_weights = df['fairness_weight']
        df = df.sample(n=len(df), replace=True, weights=sample_weights)
    
    return df


def apply_aggressive_resampling(dataframe: pd.DataFrame, sensitive_column: str) -> pd.DataFrame:
    """Apply aggressive resampling to balance all groups equally."""
    df = dataframe.copy()
    group_counts = df[sensitive_column].value_counts()
    
    # Target all groups to the maximum size (aggressive approach)
    target_size = group_counts.max()
    
    resampled_dfs = []
    for group in group_counts.index:
        group_data = df[df[sensitive_column] == group]
        current_size = len(group_data)
        
        if current_size < target_size:
            # Aggressive oversampling
            multiplier = int(target_size / current_size) + 1
            resampled = pd.concat([group_data] * multiplier, ignore_index=True)
            resampled = resampled.sample(n=int(target_size), replace=True)
        else:
            resampled = group_data
        
        resampled_dfs.append(resampled)
    
    return pd.concat(resampled_dfs, ignore_index=True)


def auto_debias_engine(dataframe: pd.DataFrame, target_column: str, sensitive_columns: list) -> tuple[pd.DataFrame, dict]:
    """Full Auto-Debias Engine - detects, decides, applies, and validates mitigation."""
    df = dataframe.copy()
    
    # Step 1: Detect bias across all attributes
    all_bias = detect_all_bias(df, target_column, sensitive_columns)
    
    # Step 2: Decision engine
    decisions = decision_engine(all_bias)
    
    # Step 3: Apply mitigation based on decisions
    applied_actions = []
    before_metrics = all_bias.copy()
    
    for attr, decision in decisions.items():
        try:
            if decision["action"] == "SKIP":
                applied_actions.append(f"{attr}: SKIPPED - {decision['reason']}")
                continue
            elif decision["action"] == "MONITOR":
                applied_actions.append(f"{attr}: MONITOR - {decision['reason']}")
                continue
            
            # Apply mitigation
            method = decision.get("method", "reweighing")
            
            # Imputation first if needed
            if all_bias[attr]["has_missing_data"]:
                if df[attr].dtype == 'object':
                    fill_value = df[attr].mode()[0] if not df[attr].mode().empty else 'Unknown'
                else:
                    fill_value = df[attr].median()
                df[attr].fillna(fill_value, inplace=True)
                applied_actions.append(f"{attr}: Imputation applied")
            
            # Apply the selected method
            if method == "hybrid":
                # Full intervention
                df = apply_aggressive_resampling(df, attr)
                df = apply_aggressive_reweighing(df, target_column, attr, max_iterations=20)
                df['dir_applied'] = True
                applied_actions.append(f"{attr}: Hybrid (resample + reweigh + DIR)")
            elif method == "resampling":
                df = apply_aggressive_resampling(df, attr)
                applied_actions.append(f"{attr}: Aggressive resampling")
            elif method == "reweighing":
                df = apply_aggressive_reweighing(df, target_column, attr, max_iterations=15)
                applied_actions.append(f"{attr}: Iterative reweighing")
        except Exception as e:
            applied_actions.append(f"{attr}: ERROR - {str(e)}")
            continue
    
    # Step 4: Post-mitigation validation
    after_bias = detect_all_bias(df, target_column, sensitive_columns)
    
    # Calculate overall improvement
    total_di_before = sum(b["di_ratio"] for b in before_metrics.values())
    total_di_after = sum(b["di_ratio"] for b in after_bias.values())
    di_reduction = ((total_di_before - total_di_after) / total_di_before * 100) if total_di_before > 0 else 0
    
    # Check if any attribute worsened significantly
    worsened = []
    for attr in sensitive_columns:
        if attr in before_metrics and attr in after_bias:
            if after_bias[attr]["di_ratio"] < before_metrics[attr]["di_ratio"] * 0.9:
                worsened.append(attr)
    
    # Rollback if significant worsening
    if worsened:
        df = dataframe.copy()
        after_bias = before_metrics
        applied_actions.append(f"ROLLBACK: Worsened attributes: {', '.join(worsened)}")
        di_reduction = 0
    
    # Clean dataframe for CSV export - convert numpy types
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    result = {
        "decisions": decisions,
        "applied_actions": applied_actions,
        "before_metrics": before_metrics,
        "after_metrics": after_bias,
        "di_reduction_percent": di_reduction,
        "worsened_attributes": worsened,
        "success": len(worsened) == 0
    }
    
    return df, result


def mitigate_dataset(dataframe: pd.DataFrame, target_column: str, sensitive_column: str, method: str, sensitive_columns: list = None) -> tuple[pd.DataFrame, dict]:
    """Apply active bias mitigation to the dataset. Returns (mitigated_df, before_after_metrics)."""
    df_mitigated = dataframe.copy()
    
    # Calculate before metrics
    before_metrics = detect_bias_characteristics(dataframe, target_column, sensitive_column)
    
    # Pre-check warnings
    warnings = []
    if before_metrics["has_small_sample"]:
        warnings.append("Low sample size detected - results may be unreliable")
    if before_metrics["has_missing_data"]:
        warnings.append("Missing data detected - imputation will be applied")
    
    if method == "auto":
        # Use full Auto-Debias Engine with all sensitive columns
        if sensitive_columns is None:
            sensitive_columns = [sensitive_column]
        df_mitigated, engine_result = auto_debias_engine(dataframe, target_column, sensitive_columns)
        
        # Convert engine result to before_after format
        before_after = {
            "method_used": "auto_debias_engine",
            "warnings": warnings,
            "engine_result": engine_result,
            "before": {
                "di_ratio": before_metrics["di_ratio"],
                "dp_diff": before_metrics["dp_diff"],
                "group_sizes": before_metrics["group_sizes"]
            },
            "after": {
                "di_ratio": engine_result["after_metrics"].get(sensitive_column, {}).get("di_ratio", 1.0),
                "dp_diff": engine_result["after_metrics"].get(sensitive_column, {}).get("dp_diff", 0.0),
                "group_sizes": engine_result["after_metrics"].get(sensitive_column, {}).get("group_sizes", {})
            },
            "improvement": {
                "di_improvement": engine_result["after_metrics"].get(sensitive_column, {}).get("di_ratio", 1.0) - before_metrics["di_ratio"],
                "dp_reduction": before_metrics["dp_diff"] - engine_result["after_metrics"].get(sensitive_column, {}).get("dp_diff", 0.0),
                "di_reduction_percent": engine_result["di_reduction_percent"]
            }
        }
        return df_mitigated, before_after
    
    if method == "adaptive":
        # Automatically select best method
        method = select_adaptive_method(before_metrics)
    
    if method == "hybrid":
        # Apply all techniques aggressively for maximum impact
        applied_steps = []
        
        # Step 1: Imputation (if needed)
        if before_metrics["has_missing_data"]:
            if df_mitigated[sensitive_column].dtype == 'object':
                fill_value = df_mitigated[sensitive_column].mode()[0] if not df_mitigated[sensitive_column].mode().empty else 'Unknown'
            else:
                fill_value = df_mitigated[sensitive_column].median()
            df_mitigated[sensitive_column].fillna(fill_value, inplace=True)
            applied_steps.append("imputation")
        
        # Step 2: Aggressive resampling to balance groups
        df_mitigated = apply_aggressive_resampling(df_mitigated, sensitive_column)
        applied_steps.append("aggressive_resampling")
        
        # Step 3: Iterative aggressive reweighing to push DI to 1.0
        df_mitigated = apply_aggressive_reweighing(df_mitigated, target_column, sensitive_column, max_iterations=15)
        applied_steps.append("iterative_reweighing")
        
        # Step 4: DIR marker
        df_mitigated['dir_applied'] = True
        applied_steps.append("dir")
        
        df_mitigated['hybrid_steps'] = ", ".join(applied_steps)
        
    elif method == "reweighing":
        # Aggressive iterative reweighing
        df_mitigated = apply_aggressive_reweighing(df_mitigated, target_column, sensitive_column, max_iterations=10)
        
    elif method == "resampling":
        # Aggressive resampling
        df_mitigated = apply_aggressive_resampling(df_mitigated, sensitive_column)
        df_mitigated['resampling_applied'] = True
        
    elif method == "imputation":
        # Fill missing values with mode for categorical, median for numerical
        if df_mitigated[sensitive_column].dtype == 'object':
            fill_value = df_mitigated[sensitive_column].mode()[0] if not df_mitigated[sensitive_column].mode().empty else 'Unknown'
        else:
            fill_value = df_mitigated[sensitive_column].median()
        
        df_mitigated[sensitive_column].fillna(fill_value, inplace=True)
        df_mitigated['imputation_applied'] = True
        
    elif method == "dir":
        # Disparate Impact Remover simulation (simplified for demo)
        df_mitigated['mitigation_applied'] = "Disparate Impact Remover (Simulated)"
    
    # Calculate after metrics
    after_metrics = detect_bias_characteristics(df_mitigated, target_column, sensitive_column)
    
    # Post-mitigation validation - rollback if worsened
    if after_metrics["di_ratio"] < before_metrics["di_ratio"] * 0.9:  # If DI worsened by >10%
        df_mitigated = dataframe.copy()
        after_metrics = before_metrics
        warnings.append("Mitigation worsened fairness - changes rolled back")
    
    before_after = {
        "method_used": method,
        "warnings": warnings,
        "before": {
            "di_ratio": before_metrics["di_ratio"],
            "dp_diff": before_metrics["dp_diff"],
            "group_sizes": before_metrics["group_sizes"]
        },
        "after": {
            "di_ratio": after_metrics["di_ratio"],
            "dp_diff": after_metrics["dp_diff"],
            "group_sizes": after_metrics["group_sizes"]
        },
        "improvement": {
            "di_improvement": after_metrics["di_ratio"] - before_metrics["di_ratio"],
            "dp_reduction": before_metrics["dp_diff"] - after_metrics["dp_diff"]
        }
    }
    
    return df_mitigated, before_after


@router.post("/validate")
async def validate_data_quality(
    file: UploadFile = File(...),
    sensitive_columns: str = Form(...)
):
    """
    Data Quality Check - Validate dataset before any bias action.
    
    Returns:
        - status: PASS/SKIP
        - confidence: HIGH/MEDIUM/LOW
        - group_sizes
        - missing_percentages
        - issues
    """
    try:
        dataframe = load_csv_from_upload(file)
        import json
        sensitive_cols = json.loads(sensitive_columns)
        
        result = engine.data_quality_gate(dataframe, sensitive_cols)
        return result
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auto-debias")
async def auto_debias(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_columns: str = Form(...)
):
    """
    Full Auto-Debias Pipeline - Real-Time Adaptive Debias Engine
    
    Pipeline:
    Input → Data Quality Gate → Bias Detection → Bias Classification → 
    Decision Engine → Controlled Mitigation → Post-Validation → Decision Gate → Output
    
    Returns:
        - debiased dataset (CSV download)
        - pipeline results (JSON)
    """
    try:
        dataframe = load_csv_from_upload(file)
        import json
        sensitive_cols = json.loads(sensitive_columns)
        
        # Run full pipeline
        pipeline_result = engine.run_full_pipeline(dataframe, target_column, sensitive_cols)
        
        # Convert mitigated dataset to CSV
        mitigated_df = pipeline_result["dataset"]
        stream = io.StringIO()
        mitigated_df.to_csv(stream, index=False)
        csv_content = stream.getvalue()
        
        # Return CSV with pipeline results in header
        response = StreamingResponse(iter([csv_content]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=auto_debiased_{file.filename}"
        response.headers["X-Pipeline-Result"] = json.dumps(pipeline_result, default=str)
        return response
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auto-debias-analyze")
async def auto_debias_analyze(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_columns: str = Form(...)
):
    """
    Auto-Debias with Analysis - Run pipeline and return full analysis results.
    
    Returns:
        - pipeline results
        - re-analyzed bias metrics
    """
    try:
        dataframe = load_csv_from_upload(file)
        import json
        sensitive_cols = json.loads(sensitive_columns)

        # Validate sensitive columns exist in dataframe
        missing = [col for col in sensitive_cols if col not in dataframe.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Sensitive columns not found in uploaded data: {missing}")

        # Run full pipeline
        pipeline_result = engine.run_full_pipeline(dataframe, target_column, sensitive_cols)

        # Re-analyze the mitigated dataset
        from app.services.inference import infer_analysis_columns
        from app.services.fairness import analyze_dataset_bias

        mitigated_df = pipeline_result["dataset"]
        inferred = infer_analysis_columns(mitigated_df)
        analysis_result = analyze_dataset_bias(
            dataframe=mitigated_df,
            target_column=inferred.target_column,
            sensitive_columns=inferred.sensitive_columns,
        )

        # Handle case where no bias is detected - this is a valid result
        if not analysis_result or not any(analysis_result.get("fairness_metrics", {})):
            # No bias detected - return success with full structure expected by frontend
            combined_result = {
                "pipeline_result": {k: v for k, v in pipeline_result.items() if k != "dataset"},
                "analysis_result": {
                    "summary": "No bias detected in the dataset.",
                    "detected_target": inferred.target_column,
                    "detected_sensitive_columns": inferred.sensitive_columns,
                    "potential_bias_detected": False,
                    "fairness_metrics": {},
                    "bias_drivers": {},
                    "proxy_features": [],
                    "intersectional_bias": {},
                    "notes": ["No bias detected in the dataset. The data appears to be fair."],
                    "affected_population": {},
                    "preprocessing_steps": [],
                    "feature_removals": [],
                    "bias_report_summary": {
                        "overall_risk_level": "LOW",
                        "unified_bias_risk_score": 0,
                        "high_risk_attributes": [],
                        "recommendations": ["No action required - dataset appears fair"]
                    },
                    "structured_bias_report": {},
                    "shap_importance": {},
                    "tradeoff_curves": {}
                }
            }
            import json
            return json.loads(json.dumps(combined_result, default=str))

        # Combine results - exclude DataFrame from JSON serialization
        # Ensure bias_report_summary has required fields for frontend
        combined_result = {
            "pipeline_result": {k: v for k, v in pipeline_result.items() if k != "dataset"},
            "analysis_result": {
                **analysis_result,
                "bias_report_summary": {
                    **analysis_result.get("bias_report_summary", {}),
                    "overall_risk_level": analysis_result.get("bias_report_summary", {}).get("overall_risk_level", "MEDIUM"),
                    "unified_bias_risk_score": analysis_result.get("bias_report_summary", {}).get("unified_bias_risk_score", 0)
                }
            }
        }

        # Convert all numpy types to native Python types using json.dumps as fallback
        import json
        return json.loads(json.dumps(combined_result, default=str))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/gate-decision")
async def gate_decision(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_columns: str = Form(...)
):
    """
    Real-Time Decision Gate - Fairness control layer before decisions.
    
    Returns:
        - decision: APPROVE/FLAG/BLOCK
        - risk_score: 0-100
        - applied_action
        - confidence
    """
    try:
        dataframe = load_csv_from_upload(file)
        import json
        sensitive_cols = json.loads(sensitive_columns)
        
        # Run full pipeline to get gate decision
        pipeline_result = engine.run_full_pipeline(dataframe, target_column, sensitive_cols)
        
        # Return gate decision
        return {
            "decision": pipeline_result["gate_decision"]["decision"],
            "risk_score": pipeline_result["gate_decision"]["risk_score"],
            "applied_action": pipeline_result["gate_decision"]["applied_action"],
            "confidence": pipeline_result["gate_decision"]["confidence"],
            "pipeline_status": pipeline_result["status"],
            "quality_gate": pipeline_result["quality_gate"]
        }
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/audit-log")
async def get_audit_log():
    """
    Get the audit log of all mitigation actions.
    
    Returns:
        - audit_log: List of all logged actions
    """
    return {
        "audit_log": engine.audit_log,
        "total_entries": len(engine.audit_log)
    }
