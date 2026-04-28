import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeDecisions, detectColumns } from "../api";

const METRIC_INFO = {
  demographic_parity: {
    label: "Demographic Parity",
    icon: "⚖",
    desc: "Are positive predictions equally likely across groups?",
  },
  equalized_odds: {
    label: "Equalized Odds",
    icon: "🎯",
    desc: "Do groups have equal True Positive and False Positive rates?",
  },
  equal_opportunity: {
    label: "Equal Opportunity",
    icon: "🤝",
    desc: "Are qualified individuals equally likely to receive positive decisions?",
  },
  predictive_parity: {
    label: "Predictive Parity",
    icon: "📊",
    desc: "Is precision equal across groups?",
  },
  accuracy_parity: {
    label: "Accuracy Parity",
    icon: "✓",
    desc: "Is the model equally accurate across groups?",
  },
  false_negative_rate_parity: {
    label: "FNR Parity",
    icon: "⚠",
    desc: "Are qualified individuals equally likely to be wrongly rejected?",
  },
};

function MetricCard({ metricKey, metric }) {
  const info = METRIC_INFO[metricKey] || { label: metricKey, icon: "•", desc: "" };
  const severity = metric.severity || "LOW";
  const passed = metric.passed;

  const mainValue =
    metric.diff !== undefined ? `${(metric.diff * 100).toFixed(1)}%` :
    metric.combined_diff !== undefined ? `${(metric.combined_diff * 100).toFixed(1)}%` :
    "—";

  return (
    <div className={`metricCard metric-${severity.toLowerCase()} ${passed ? "metric-passed" : "metric-failed"}`}>
      <div className="metricCardHeader">
        <span className="metricIcon">{info.icon}</span>
        <div>
          <strong className="metricLabel">{info.label}</strong>
          <span className={`metricBadge ${passed ? "badge-pass" : "badge-fail"}`}>
            {passed ? "PASS" : "FAIL"}
          </span>
        </div>
        <span className={`severityTag sev-${severity.toLowerCase()}`}>{severity}</span>
      </div>
      <p className="metricDesc">{info.desc}</p>
      <div className="metricValue">{mainValue} disparity</div>
      {metric.highest_group && metric.lowest_group && (
        <div className="metricGroups">
          <span className="groupHigh">↑ {metric.highest_group}</span>
          <span className="groupLow">↓ {metric.lowest_group}</span>
        </div>
      )}
      {metricKey === "equalized_odds" && (
        <div className="metricSubValues">
          <span>TPR diff: {(metric.tpr_diff * 100).toFixed(1)}%</span>
          <span>FPR diff: {(metric.fpr_diff * 100).toFixed(1)}%</span>
        </div>
      )}
    </div>
  );
}

function GroupStatsTable({ groupStats }) {
  const groups = Object.entries(groupStats);
  if (!groups.length) return null;

  return (
    <div className="groupStatsTable">
      <table>
        <thead>
          <tr>
            <th>Group</th>
            <th>N</th>
            <th>Accuracy</th>
            <th>Selection Rate</th>
            <th>TPR</th>
            <th>FPR</th>
            <th>FNR</th>
            <th>Precision</th>
          </tr>
        </thead>
        <tbody>
          {groups.map(([group, stats]) => (
            <tr key={group}>
              <td><strong>{group}</strong></td>
              <td>{stats.n}</td>
              <td>{(stats.accuracy * 100).toFixed(1)}%</td>
              <td>{(stats.selection_rate * 100).toFixed(1)}%</td>
              <td className={stats.tpr < 0.5 ? "low-val" : ""}>{(stats.tpr * 100).toFixed(1)}%</td>
              <td className={stats.fpr > 0.3 ? "high-val" : ""}>{(stats.fpr * 100).toFixed(1)}%</td>
              <td className={stats.fnr > 0.3 ? "high-val" : ""}>{(stats.fnr * 100).toFixed(1)}%</td>
              <td>{(stats.precision * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function DecisionAnalysis() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [detected, setDetected] = useState(null);
  const [predCol, setPredCol] = useState("");
  const [actualCol, setActualCol] = useState("");
  const [sensColsInput, setSensColsInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [expandedAttr, setExpandedAttr] = useState(null);

  const handleFileChange = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
    setDetected(null);

    // Auto-detect columns
    setDetecting(true);
    try {
      const det = await detectColumns({ file: f });
      setDetected(det);
      if (det.detected.prediction_column) setPredCol(det.detected.prediction_column);
      if (det.detected.actual_column) setActualCol(det.detected.actual_column);
      if (det.detected.sensitive_columns?.length) {
        setSensColsInput(det.detected.sensitive_columns.join(", "));
      }
    } catch {
      // silently ignore — user can fill manually
    } finally {
      setDetecting(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError("Please upload a predictions CSV."); return; }
    if (!predCol.trim()) { setError("Please specify the prediction column."); return; }
    if (!actualCol.trim()) { setError("Please specify the actual outcome column."); return; }
    if (!sensColsInput.trim()) { setError("Please specify at least one sensitive attribute column."); return; }

    setLoading(true);
    setError("");
    try {
      const data = await analyzeDecisions({
        file,
        predictionColumn: predCol.trim(),
        actualColumn: actualCol.trim(),
        sensitiveColumns: sensColsInput.trim(),
      });
      setResult(data);
      setExpandedAttr(data.sensitive_attributes_analyzed?.[0] || null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="workspace animate-fadeIn">
      <header className="workspaceHeader animate-slideUp">
        <button className="backButton" onClick={() => navigate("/")}>← Back</button>
        <div>
          <p className="eyebrow">FairSight Core</p>
          <h1>AI Decision Fairness Analysis</h1>
          <p className="subtitle">
            Upload your model's predictions to detect bias in automated decisions
          </p>
        </div>
      </header>

      {/* What this does */}
      <div className="decisionInfoBanner animate-slideUp delay-1">
        <div className="infoItem">
          <span className="infoIcon">🤖</span>
          <div>
            <strong>Model Predictions</strong>
            <p>Upload a CSV with your AI model's outputs</p>
          </div>
        </div>
        <div className="infoArrow">→</div>
        <div className="infoItem">
          <span className="infoIcon">⚖</span>
          <div>
            <strong>6 Fairness Metrics</strong>
            <p>Demographic Parity, Equalized Odds, Equal Opportunity + more</p>
          </div>
        </div>
        <div className="infoArrow">→</div>
        <div className="infoItem">
          <span className="infoIcon">📋</span>
          <div>
            <strong>Actionable Report</strong>
            <p>Per-group bias breakdown with mitigation steps</p>
          </div>
        </div>
      </div>

      <div className="analysisSection">
        <div className="uploadCard animate-slideUp delay-1">
          <h2>Upload Model Predictions</h2>
          <p className="uploadHint">
            CSV must contain: <strong>prediction column</strong>, <strong>actual outcome column</strong>,
            and <strong>sensitive attribute columns</strong> (gender, race, age, etc.)
          </p>

          {/* CSV format example */}
          <div className="csvFormatExample">
            <code>gender, age_group, actual, prediction</code>
            <code>Male, 25-34, 1, 1</code>
            <code>Female, 25-34, 1, 0  ← wrongly rejected</code>
            <code>Male, 45-54, 0, 0</code>
          </div>

          <form onSubmit={handleSubmit}>
            <div
              className={`dropZone ${file ? "hasFile" : ""}`}
              onClick={() => document.getElementById("decisionFileInput").click()}
            >
              <input
                id="decisionFileInput"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
              {file ? (
                <div className="filePreview">
                  <span className="fileIcon">📄</span>
                  <span className="fileName">{file.name}</span>
                  {detected && (
                    <span className="fileSize">{detected.row_count?.toLocaleString()} rows · {detected.columns?.length} columns</span>
                  )}
                </div>
              ) : (
                <div className="dropZoneContent">
                  <span className="uploadIcon">⬆</span>
                  <p>Click to upload predictions CSV</p>
                  <small>Columns auto-detected on upload</small>
                </div>
              )}
            </div>

            {detecting && <p className="detectingMsg">🔍 Auto-detecting columns...</p>}

            {detected && (
              <div className="columnConfig">
                <h4>Column Configuration</h4>
                <div className="columnGrid">
                  <div className="columnField">
                    <label>Prediction Column *</label>
                    <select value={predCol} onChange={e => setPredCol(e.target.value)}>
                      <option value="">— select —</option>
                      {detected.columns.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div className="columnField">
                    <label>Actual Outcome Column *</label>
                    <select value={actualCol} onChange={e => setActualCol(e.target.value)}>
                      <option value="">— select —</option>
                      {detected.columns.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div className="columnField fullWidth">
                    <label>Sensitive Attribute Columns * <span className="hint">(comma-separated)</span></label>
                    <input
                      type="text"
                      value={sensColsInput}
                      onChange={e => setSensColsInput(e.target.value)}
                      placeholder="e.g. gender, race, age_group"
                    />
                  </div>
                </div>
              </div>
            )}

            {error && <div className="errorBox">{error}</div>}

            <button className="primaryButton" type="submit" disabled={loading || !file}>
              {loading ? "Analyzing decisions..." : "Analyze for Bias"}
            </button>
          </form>
        </div>

        {/* Results */}
        {result && (
          <div className="decisionResults animate-slideUp delay-1">

            {/* Overall verdict banner */}
            <div className={`verdictBanner verdict-${result.overall_risk_level.toLowerCase()}`}>
              <div className="verdictMain">
                <span className="verdictIcon">
                  {result.overall_verdict === "FAIR" ? "✓" :
                   result.overall_verdict === "BIASED" ? "✗" : "⚠"}
                </span>
                <div>
                  <h2 className="verdictTitle">{result.overall_verdict}</h2>
                  <p className="verdictSummary">{result.overall_summary}</p>
                </div>
              </div>
              <div className="verdictStats">
                <div className="vStat">
                  <span className="vStatVal">{result.total_records?.toLocaleString()}</span>
                  <span className="vStatLabel">Decisions Analyzed</span>
                </div>
                <div className="vStat">
                  <span className="vStatVal">{(result.overall_accuracy * 100).toFixed(1)}%</span>
                  <span className="vStatLabel">Model Accuracy</span>
                </div>
                <div className="vStat">
                  <span className={`vStatVal risk-${result.overall_risk_level.toLowerCase()}`}>
                    {result.overall_risk_level}
                  </span>
                  <span className="vStatLabel">Bias Risk</span>
                </div>
                <div className="vStat">
                  <span className={`vStatVal ${result.compliance?.us_eeoc_80_rule ? "text-green" : "text-red"}`}>
                    {result.compliance?.us_eeoc_80_rule ? "PASS" : "FAIL"}
                  </span>
                  <span className="vStatLabel">EEOC 80% Rule</span>
                </div>
              </div>
            </div>

            {/* Top recommendations */}
            {result.top_recommendations?.length > 0 && (
              <div className="insightCard animate-slideUp">
                <h3>🔧 Top Mitigation Actions</h3>
                <div className="recList">
                  {result.top_recommendations.map((rec, i) => (
                    <div key={i} className={`recItem rec-${rec.priority.toLowerCase()}`}>
                      <div className="recHeader">
                        <span className={`recPriorityBadge priority-${rec.priority.toLowerCase()}`}>
                          {rec.priority}
                        </span>
                        <strong>{rec.action}</strong>
                        <span className="recTechnique">{rec.technique}</span>
                      </div>
                      <p>{rec.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Per-attribute analysis */}
            {Object.entries(result.per_attribute).map(([attr, data]) => (
              <div key={attr} className="insightCard animate-slideUp">
                <div
                  className="cardHeader"
                  onClick={() => setExpandedAttr(expandedAttr === attr ? null : attr)}
                  style={{ cursor: "pointer" }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <h3>{attr}</h3>
                    <span className={`riskBadge risk-${data.verdict.risk_level.toLowerCase()}`}>
                      {data.verdict.verdict}
                    </span>
                    <span className="smallMuted">
                      {data.verdict.criteria_passed}/{data.verdict.criteria_total} criteria passed
                    </span>
                  </div>
                  <span className="toggleIcon">{expandedAttr === attr ? "▲" : "▼"}</span>
                </div>

                {expandedAttr === attr && (
                  <div className="cardContent">
                    <p className="verdictSummary" style={{ marginBottom: "20px" }}>
                      {data.verdict.summary}
                    </p>

                    {/* 6 fairness metric cards */}
                    <div className="metricsGrid6">
                      {Object.entries(data.fairness_metrics).map(([key, metric]) => (
                        <MetricCard key={key} metricKey={key} metric={metric} />
                      ))}
                    </div>

                    {/* Per-group confusion matrix */}
                    <h4 style={{ marginTop: "24px", marginBottom: "12px" }}>
                      Per-Group Decision Statistics
                    </h4>
                    <GroupStatsTable groupStats={data.group_stats} />

                    {/* Attribute-level recommendations */}
                    {data.recommendations?.length > 0 && (
                      <div style={{ marginTop: "20px" }}>
                        <h4>Recommendations for "{attr}"</h4>
                        {data.recommendations.map((rec, i) => (
                          <div key={i} className={`recItem rec-${rec.priority.toLowerCase()}`} style={{ marginTop: "8px" }}>
                            <div className="recHeader">
                              <span className={`recPriorityBadge priority-${rec.priority.toLowerCase()}`}>
                                {rec.priority}
                              </span>
                              <strong>{rec.action}</strong>
                            </div>
                            <p>{rec.detail}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Compliance */}
            <div className="insightCard animate-slideUp">
              <h3>📋 Compliance Status</h3>
              <div className="complianceGrid">
                <div className={`complianceItem ${result.compliance?.eu_ai_act === "COMPLIANT" ? "comp-pass" : "comp-fail"}`}>
                  <strong>EU AI Act</strong>
                  <span>{result.compliance?.eu_ai_act?.replace("_", " ")}</span>
                </div>
                <div className={`complianceItem ${result.compliance?.us_eeoc_80_rule ? "comp-pass" : "comp-fail"}`}>
                  <strong>US EEOC 80% Rule</strong>
                  <span>{result.compliance?.us_eeoc_80_rule ? "COMPLIANT" : "NON-COMPLIANT"}</span>
                </div>
              </div>
              <p className="complianceNote">{result.compliance?.notes}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
