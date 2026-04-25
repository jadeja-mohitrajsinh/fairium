from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BiasDriver(BaseModel):
    feature: str
    impact: float = Field(..., ge=0.0)


class ProxyFeature(BaseModel):
    sensitive_column: str
    feature: str
    correlation: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class GroupAnalysis(BaseModel):
    group: str
    total_samples: int
    positive_outcomes: int
    selection_rate: float
    reliability: str


class SimulationResult(BaseModel):
    dp_diff_reduced: float
    improvement: str
    explanation: str


class FairnessMetricResult(BaseModel):
    dp_diff: float = Field(..., ge=0.0)
    di_ratio: float = Field(..., ge=0.0)
    group_rates: Dict[str, float]
    explanation: str
    severity: str
    group_analysis: List[GroupAnalysis]
    confidence: str
    confidence_explanation: str
    recommendation: str
    simulation: Optional[SimulationResult] = None


class IntersectionalBias(BaseModel):
    group: str
    selection_rate: float
    risk_level: str


class AffectedPopulation(BaseModel):
    total_affected_individuals: int
    affected_groups: List[Dict]
    explanation: str


class PreprocessingStep(BaseModel):
    type: str
    column: str
    issue: str
    recommendation: str
    priority: str


class FeatureRemoval(BaseModel):
    feature: str
    sensitive_column: str
    correlation: float
    rationale: str
    priority: str


class BiasReportSummary(BaseModel):
    overall_risk_level: str
    total_sensitive_attributes_analyzed: int
    total_records_analyzed: int
    high_risk_attributes: List[str]
    moderate_risk_attributes: List[str]
    low_risk_attributes: List[str]
    compliance_status: str
    recommendation_summary: str
    key_issue: str = ""


class AnalysisResponse(BaseModel):
    summary: str
    detected_target: str
    detected_sensitive_columns: List[str]
    potential_bias_detected: bool
    fairness_metrics: Dict[str, FairnessMetricResult]
    bias_drivers: List[BiasDriver]
    proxy_features: List[ProxyFeature]
    intersectional_bias: List[IntersectionalBias]
    notes: List[str]
    affected_population: Dict[str, AffectedPopulation]
    preprocessing_steps: List[PreprocessingStep]
    feature_removals: List[FeatureRemoval]
    bias_report_summary: BiasReportSummary
    structured_bias_report: Dict = {}


class TextBiasResult(BaseModel):
    type: str
    confidence: str
    explanation: str
    alternatives: List[str]
    keyword_matches: List[str]
    phrase_matches: List[str]
    ambiguous_matches: List[str]


class TextBiasAnalysisResponse(BaseModel):
    bias_detected: str  # Can be "Yes", "No", or "Possible"
    biases: List[TextBiasResult]
    overall_confidence: str
    ml_confidence: Optional[float] = None
    summary: str
