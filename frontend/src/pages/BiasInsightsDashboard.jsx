import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, LineChart, Line } from "recharts";
import FairnessCard from "../components/FairnessCard";
import MitigationModal from "../components/MitigationModal";
import ExecutiveSummary from "../components/ExecutiveSummary";
import PriorityActions from "../components/PriorityActions";
import TextAnalysisResult from "../components/TextAnalysisResult";

export default function BiasInsightsDashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, type } = location.state || {};
  const [expandedSections, setExpandedSections] = useState({});
  const [showMitigationModal, setShowMitigationModal] = useState(false);
  const [tradeoffLevel, setTradeoffLevel] = useState(0);

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  if (!result) {
    return (
      <div className="dashboard animate-fadeIn">
        <div className="emptyState animate-slideUp">
          <h2>No Analysis Results</h2>
          <p>Please run an analysis first to view insights.</p>
          <button className="primaryButton" onClick={() => navigate("/")}>
            Go to Analysis Workspace
          </button>
        </div>
      </div>
    );
  }

  if (type === "text") {
    return (
      <div className="dashboard animate-fadeIn">
        <header className="dashboardHeader animate-slideUp">
          <button className="backButton" onClick={() => navigate("/")}>
            Back to Workspace
          </button>
          <div>
            <p className="eyebrow">Text Bias Analysis</p>
            <h1>Bias Insights</h1>
          </div>
        </header>
        <TextAnalysisResult result={result} />
      </div>
    );
  }

  const riskLevel = result.bias_report_summary.overall_risk_level;
  const fairnessMetrics = result.fairness_metrics;

  return (
    <div className="dashboard animate-fadeIn">
      <header className="dashboardHeader animate-slideUp">
        <button className="backButton" onClick={() => navigate("/")}>
          Back to Workspace
        </button>
        <div>
          <p className="eyebrow">Dataset Bias Analysis</p>
          <h1>Bias Insights Dashboard</h1>
        </div>
        <button className="primaryButton" onClick={() => setShowMitigationModal(true)} style={{marginLeft: 'auto'}}>
          Debias Dataset
        </button>
      </header>

      <div className="dashboardContent">
        <ExecutiveSummary report={result.structured_bias_report} />
        
        <PriorityActions recommendations={result.structured_bias_report?.recommendations} />

        {/* SHAP Chart */}
        {result.shap_importance && result.shap_importance.length > 0 && (
          <div className="insightCard animate-slideUp delay-2">
            <h3>Explainable AI (SHAP) & Proxy Detection</h3>
            <div className="chartContainer" style={{ height: "300px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.shap_importance} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#111111" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            {result.proxy_features && result.proxy_features.length > 0 && (
              <div className="proxyWarnings" style={{marginTop: '20px'}}>
                <h4>Proxy Feature Warnings</h4>
                {result.proxy_features.map((proxy, idx) => (
                  <div key={idx} className="priorityAction urgent" style={{padding: '10px', marginTop: '10px'}}>
                    <strong>{proxy.feature}</strong> acts as a proxy for <strong>{proxy.sensitive_column}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Summary Banner */}
        <div className={`summaryBanner risk-${riskLevel.toLowerCase()} animate-slideUp delay-2`}>
          <div className="summaryItem">
            <span className="summaryLabel">Risk Level</span>
            <span className="summaryValue">{riskLevel}</span>
          </div>
          <div className="summaryItem">
            <span className="summaryLabel">Records Analyzed</span>
            <span className="summaryValue">{result.bias_report_summary.total_records_analyzed}</span>
          </div>
          <div className="summaryItem">
            <span className="summaryLabel">Sensitive Attributes</span>
            <span className="summaryValue">{result.bias_report_summary.total_sensitive_attributes_analyzed}</span>
          </div>
        </div>

        {/* Fairness Cards Grid */}
        <div className="metricsGrid animate-slideUp delay-3">
          {Object.entries(fairnessMetrics).map(([column, metric]) => (
            <FairnessCard key={column} column={column} metric={metric} />
          ))}
        </div>

        {/* Expandable Sections */}
        <div className="insightCard animate-slideUp delay-3">
          <div className="cardHeader" onClick={() => toggleSection("drivers")}>
            <h3>Bias Drivers</h3>
            <span className="toggleIcon">{expandedSections.drivers ? "v" : ">"}</span>
          </div>
          {expandedSections.drivers && (
            <div className="cardContent">
              {result.bias_drivers.map((driver, idx) => (
                <div key={idx} className="driverItem">
                  <span className="driverFeature">{driver.feature}</span>
                  <span className="driverImpact">{driver.impact.toFixed(4)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <MitigationModal 
        show={showMitigationModal} 
        onClose={() => setShowMitigationModal(false)}
        targetColumn={result.detected_target}
        sensitiveColumn={result.detected_sensitive_columns[0]}
      />
    </div>
  );
}
