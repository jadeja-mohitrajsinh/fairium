from typing import Any, Dict, List, Optional

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
    shap_importance: List[Dict] = []
    tradeoff_curves: Dict[str, List[Dict]] = {}


class TextBiasResult(BaseModel):
    type: str
    confidence: str
    explanation: str
    alternatives: List[str]
    keyword_matches: List[str]
    phrase_matches: List[str]
    ambiguous_matches: List[str]


class TextBiasRequest(BaseModel):
    text: str


class TextBiasAnalysisResponse(BaseModel):
    bias_detected: str  # Can be "Yes", "No", or "Possible"
    biases: List[TextBiasResult]
    overall_confidence: str
    ml_confidence: Optional[float] = None
    summary: str


class BiasDriftResult(BaseModel):
    detected: bool
    delta: float
    percent_change: float
    baseline_score: float
    current_score: float
    explanation: str


class BiasAlert(BaseModel):
    level: str
    message: str
    threshold: float
    current_value: float


class RiskHeatmapCell(BaseModel):
    attribute: str
    score: float
    risk_level: str


class PrioritizationItem(BaseModel):
    attribute: str
    priority_score: float
    priority_level: str
    rationale: str


class DataIntelligenceInsight(BaseModel):
    category: str
    severity: str
    finding: str
    suggestion: str


class MitigationPreviewItem(BaseModel):
    attribute: str
    method: str
    expected_risk_reduction: float
    expected_fairness_gain: str
    notes: str


class DecisionIntelligenceSummary(BaseModel):
    unified_bias_risk_score: int = Field(..., ge=0, le=100)
    aggregated_metrics: Dict[str, float]
    prioritization: List[PrioritizationItem]


class MonitorRequest(BaseModel):
    analysis_payload: Dict[str, Any]
    historical_risk_scores: List[float] = []
    thresholds: Dict[str, float] = {}
    scenario: Optional[str] = None
    external_metadata: Dict[str, Any] = {}


class MonitorResponse(BaseModel):
    generated_at: str
    decision_intelligence: DecisionIntelligenceSummary
    drift: BiasDriftResult
    alerts: List[BiasAlert]
    risk_heatmap: List[RiskHeatmapCell]
    data_intelligence: List[DataIntelligenceInsight]
    mitigation_preview: List[MitigationPreviewItem]
    explainability: List[str]
    impacted_groups: List[str]
    audit_report: Dict[str, Any]
    tracking_log_id: str


class GateRequest(BaseModel):
    decision_id: str
    scenario: str = "general"
    decision_payload: Dict[str, Any]
    analysis_payload: Dict[str, Any]
    risk_score_override: Optional[int] = Field(default=None, ge=0, le=100)
    block_threshold: int = Field(default=75, ge=0, le=100)
    flag_threshold: int = Field(default=50, ge=0, le=100)
    auto_mitigation: bool = True
    external_metadata: Dict[str, Any] = {}


class GateResponse(BaseModel):
    generated_at: str
    decision_id: str
    scenario: str
    status: str
    fairness_approval_required: bool
    fairness_approved: bool
    bias_risk_score: int = Field(..., ge=0, le=100)
    reasons: List[str]
    impacted_groups: List[str]
    risk_consequences: List[str]
    mitigation_actions: List[str]
    fairness_certificate: Dict[str, Any]
    tracking_log_id: str
