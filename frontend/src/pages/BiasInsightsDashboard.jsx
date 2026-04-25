import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, LineChart, Line } from "recharts";

export default function BiasInsightsDashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const { result, type } = location.state || {};
  const [expandedSections, setExpandedSections] = useState({});
  const [showMitigationModal, setShowMitigationModal] = useState(false);
  const [mitigationMethod, setMitigationMethod] = useState("reweighing");
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
            ← Back to Workspace
          </button>
          <div>
            <p className="eyebrow">Text Bias Analysis</p>
            <h1>Bias Insights</h1>
          </div>
        </header>

        <div className="dashboardContent">
          <div className="summaryBanner animate-slideUp delay-1">
            <div className="summaryItem">
              <span className="summaryLabel">Bias Status</span>
              <span className={`summaryValue status-${result.bias_detected.toLowerCase()}`}>
                {result.bias_detected}
              </span>
            </div>
            <div className="summaryItem">
              <span className="summaryLabel">Confidence</span>
              <span className="summaryValue">{result.overall_confidence}</span>
            </div>
            {result.ml_confidence && (
              <div className="summaryItem">
                <span className="summaryLabel">ML Confidence</span>
                <span className="summaryValue">{(result.ml_confidence * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>

          <div className="insightCard animate-slideUp delay-2">
            <h3>Summary</h3>
            <p>{result.summary}</p>
          </div>

          {result.biases.length > 0 && (
            <div className="insightCard animate-slideUp delay-3">
              <h3>Detailed Bias Analysis</h3>
              {result.biases.map((bias, idx) => (
                <div key={idx} className="biasDetail">
                  <div className="biasDetailHeader">
                    <strong>{bias.type.charAt(0).toUpperCase() + bias.type.slice(1)} Bias</strong>
                    <span className={`confidenceBadge confidence-${bias.confidence.toLowerCase()}`}>
                      {bias.confidence} Confidence
                    </span>
                  </div>
                  <p className="biasExplanation">{bias.explanation}</p>
                  {bias.alternatives.length > 0 && (
                    <div className="biasAlternatives">
                      <strong>Suggested alternatives:</strong>
                      <ul>
                        {bias.alternatives.map((alt, altIdx) => (
                          <li key={altIdx}>{alt}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Dataset bias dashboard
  const riskLevel = result.bias_report_summary.overall_risk_level;
  const fairnessMetrics = result.fairness_metrics;

  return (
    <div className="dashboard animate-fadeIn">
      <header className="dashboardHeader animate-slideUp">
        <button className="backButton" onClick={() => navigate("/")}>
          ← Back to Workspace
        </button>
        <div>
          <p className="eyebrow">Dataset Bias Analysis</p>
          <h1>Bias Insights Dashboard</h1>
        </div>
        <button className="primaryButton" onClick={() => setShowMitigationModal(true)} style={{marginLeft: 'auto'}}>
          ✨ Debias Dataset
        </button>
      </header>

      <div className="dashboardContent">
        {/* Executive Summary */}
        {result.structured_bias_report && result.structured_bias_report.overall_summary && (
          <div className="executiveSummaryCard animate-slideUp delay-1">
            <div className="executiveHeader">
              <h3>📋 Executive Summary</h3>
              <span className={`decisionLabel decision-${result.bias_report_summary.compliance_status.toLowerCase()}`}>
                {result.bias_report_summary.compliance_status === "REQUIRES_ACTION" ? "🚨 Action Required" :
                 result.bias_report_summary.compliance_status === "MONITOR" ? "⚠️ Monitor" : "✅ Safe"}
              </span>
            </div>
            <p className="executiveText">{result.structured_bias_report.overall_summary.executive_summary}</p>
            
            {/* Recommended Decision Box */}
            {result.structured_bias_report.overall_summary.recommended_decision && (
              <div className="decisionBox">
                <strong>Recommended Decision:</strong>
                <p>{result.structured_bias_report.overall_summary.recommended_decision}</p>
              </div>
            )}
            
            {/* Reliability Warning */}
            {result.structured_bias_report.overall_summary.reliability_warning && (
              <div className="reliabilityWarning">
                {result.structured_bias_report.overall_summary.reliability_warning}
              </div>
            )}
          </div>
        )}

        {/* Priority Actions - What to do next */}
        {result.structured_bias_report && result.structured_bias_report.recommendations && (
          <div className="priorityActionsCard animate-slideUp delay-1">
            <h3>🎯 Priority Actions - What to Do Next</h3>
            {result.structured_bias_report.recommendations.urgent_actions && result.structured_bias_report.recommendations.urgent_actions.length > 0 ? (
              <div className="priorityAction urgent">
                <div className="priorityHeader">
                  <span className="priorityIcon">🔴</span>
                  <strong>{result.structured_bias_report.recommendations.urgent_actions[0].action}</strong>
                </div>
                <p>{result.structured_bias_report.recommendations.urgent_actions[0].reason}</p>
                <span className="priorityTimeline">⏱️ {result.structured_bias_report.recommendations.urgent_actions[0].timeline}</span>
              </div>
            ) : result.structured_bias_report.recommendations.monitor_actions && result.structured_bias_report.recommendations.monitor_actions.length > 0 ? (
              <div className="priorityAction monitor">
                <div className="priorityHeader">
                  <span className="priorityIcon">🟡</span>
                  <strong>{result.structured_bias_report.recommendations.monitor_actions[0].action}</strong>
                </div>
                <p>{result.structured_bias_report.recommendations.monitor_actions[0].reason}</p>
                <span className="priorityTimeline">⏱️ {result.structured_bias_report.recommendations.monitor_actions[0].timeline}</span>
              </div>
            ) : (
              <div className="priorityAction safe">
                <div className="priorityHeader">
                  <span className="priorityIcon">🟢</span>
                  <strong>No immediate action required</strong>
                </div>
                <p>Continue regular monitoring and bias audits.</p>
              </div>
            )}
          </div>
        )}

        {/* Explainable AI & Proxy Detection */}
        {result.shap_importance && result.shap_importance.length > 0 && (
          <div className="insightCard animate-slideUp delay-2">
            <h3>🧠 Explainable AI (SHAP) & Proxy Detection</h3>
            <p className="cardDescription">
              True feature importance calculated using SHAP values. Features highly correlated with sensitive attributes are flagged as potential proxies.
            </p>
            <div className="chartContainer" style={{ height: "300px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={result.shap_importance} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" width={120} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#111111" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            {/* Proxy warnings */}
            {result.proxy_features && result.proxy_features.length > 0 && (
              <div className="proxyWarnings" style={{marginTop: '20px'}}>
                <h4>⚠️ Proxy Feature Warnings</h4>
                {result.proxy_features.map((proxy, idx) => (
                  <div key={idx} className="priorityAction urgent" style={{padding: '10px', marginTop: '10px'}}>
                    <strong>{proxy.feature}</strong> acts as a proxy for <strong>{proxy.sensitive_column}</strong> (Correlation: {proxy.correlation.toFixed(2)})
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

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
          <div className="summaryItem">
            <span className="summaryLabel">Compliance Status</span>
            <span className="summaryValue">{result.bias_report_summary.compliance_status}</span>
          </div>
        </div>

        <div className="metricsGrid animate-slideUp delay-3">
          {Object.entries(fairnessMetrics).map(([column, metric]) => {
            const fairnessScore = Math.round(metric.di_ratio * 100);
            const getScoreColor = (score) => {
              if (score >= 80) return '#10B981';
              if (score >= 50) return '#F59E0B';
              return '#EF4444';
            };
            
            const getTopGroups = () => {
              if (!metric.group_rates) return [];
              return Object.entries(metric.group_rates)
                .map(([group, rate]) => ({ group, rate: rate * 100 }))
                .sort((a, b) => b.rate - a.rate)
                .slice(0, 3);
            };
            
            const getLowestGroup = () => {
              if (!metric.group_rates) return null;
              const sorted = Object.entries(metric.group_rates)
                .map(([group, rate]) => ({ group, rate: rate * 100 }))
                .sort((a, b) => a.rate - b.rate);
              return sorted[0];
            };
            
            const topGroups = getTopGroups();
            const lowestGroup = getLowestGroup();
            const isLowConfidence = metric.confidence === 'LOW';
            const isHighRisk = metric.severity === 'HIGH';

            const getInsight = (severity, score) => {
              if (severity === 'HIGH' && score < 20) {
                return "Severe disparity caused by near-zero outcomes in some groups";
              } else if (severity === 'HIGH') {
                return "Significant imbalance across categories affecting fairness";
              } else if (severity === 'MODERATE') {
                return "Moderate disparity requiring monitoring and validation";
              } else {
                return "Fair outcomes across groups with minimal disparity";
              }
            };

            const truncateText = (text, maxLength = 80) => {
              if (text.length <= maxLength) return text;
              return text.substring(0, maxLength).trim() + '...';
            };

            return (
              <div key={column} className={`fairnessCard ${isHighRisk ? 'high-risk-card' : ''} ${isLowConfidence ? 'low-confidence-card' : ''}`}>
                <div className="fairnessCardHeader">
                  <h3>{column}</h3>
                  <span className={`riskBadge risk-${metric.severity.toLowerCase()}`}>
                    {metric.severity}
                  </span>
                </div>
                
                {isLowConfidence && (
                  <div className="confidenceWarning">
                    <span className="warningIcon">⚠️</span>
                    <span>Low confidence due to small sample size</span>
                  </div>
                )}
                
                <div className="fairnessScoreContainer">
                  <div className="circularProgress">
                    <svg viewBox="0 0 36 36" className="circularChart">
                      <path
                        className="circle-bg"
                        d="M18 2.0845
                           a 15.9155 15.9155 0 0 1 0 31.831
                           a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <path
                        className={`circle ${isLowConfidence ? 'faded' : ''}`}
                        stroke={getScoreColor(fairnessScore)}
                        strokeDasharray={`${fairnessScore}, 100`}
                        d="M18 2.0845
                           a 15.9155 15.9155 0 0 1 0 31.831
                           a 15.9155 15.9155 0 0 1 0 -31.831"
                      />
                      <text x="18" y="20.35" className="percentage">
                        {fairnessScore}%
                      </text>
                    </svg>
                  </div>
                  <div className="fairnessScoreLabel">
                    <span>Fairness Score (DI Ratio)</span>
                    <span className="scoreLabel">{metric.di_ratio.toFixed(3)}</span>
                  </div>
                </div>

                <div className="quickStats">
                  <div className="quickStat">
                    <span className="statLabel">DP Diff</span>
                    <span className="statValue">{(metric.dp_diff * 100).toFixed(1)}%</span>
                  </div>
                  <div className="quickStat">
                    <span className="statLabel">Confidence</span>
                    <span className="statValue">{metric.confidence}</span>
                  </div>
                </div>

                {topGroups.length > 0 && (
                  <div className="topGroupsList">
                    <span className="topGroupsLabel">Group Outcome Comparison</span>
                    {topGroups.map((group, idx) => (
                      <div key={idx} className="topGroupItem">
                        <span className="groupName">{group.group}</span>
                        <span className="groupRate">{group.rate.toFixed(1)}%</span>
                      </div>
                    ))}
                    {lowestGroup && lowestGroup.rate < 50 && (
                      <div className="topGroupItem lowest-group">
                        <span className="groupName">{lowestGroup.group}</span>
                        <span className="groupRate low-rate">{lowestGroup.rate.toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                )}

                <div className="fairnessInsight">
                  <p>{getInsight(metric.severity, fairnessScore)}</p>
                </div>

                <div className="fairnessExplanation">
                  <p>{truncateText(metric.explanation)}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="insightCard animate-slideUp delay-3">
          <div className="cardHeader" onClick={() => toggleSection("drivers")}>
            <h3>Bias Drivers</h3>
            <span className="toggleIcon">{expandedSections.drivers ? "▼" : "▶"}</span>
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

        {result.intersectional_bias && result.intersectional_bias.length > 0 && (
          <div className="insightCard animate-slideUp delay-3">
            <div className="cardHeader" onClick={() => toggleSection("intersectional")}>
              <h3>Intersectional Bias</h3>
              <span className="toggleIcon">{expandedSections.intersectional ? "▼" : "▶"}</span>
            </div>
            {expandedSections.intersectional && (
              <div className="cardContent">
                <p className="intersectionalSummary">
                  {result.intersectional_bias.length > 3 
                    ? "Certain attribute combinations show consistently low outcomes across intersectional groups."
                    : "Intersectional analysis reveals disparities in specific attribute combinations."
                  }
                </p>
                {result.intersectional_bias.slice(0, 3).map((bias, idx) => (
                  <div key={idx} className="intersectionalItem">
                    <strong>{bias.group}</strong>
                    <p>Selection Rate: {(bias.selection_rate * 100).toFixed(1)}%</p>
                    <p>Risk Level: {bias.risk_level}</p>
                  </div>
                ))}
                {result.intersectional_bias.length > 3 && (
                  <button 
                    className="viewAllButton"
                    onClick={(e) => {
                      e.stopPropagation();
                      const element = e.target.closest('.cardContent');
                      element.classList.toggle('showAll');
                    }}
                  >
                    View all ({result.intersectional_bias.length} combinations)
                  </button>
                )}
                {result.intersectional_bias.length > 3 && (
                  <div className="remainingGroups">
                    {result.intersectional_bias.slice(3).map((bias, idx) => (
                      <div key={idx + 3} className="intersectionalItem">
                        <strong>{bias.group}</strong>
                        <p>Selection Rate: {(bias.selection_rate * 100).toFixed(1)}%</p>
                        <p>Risk Level: {bias.risk_level}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <div className="insightCard animate-slideUp delay-3">
          <div className="cardHeader" onClick={() => toggleSection("impact")}>
            <h3>Impact Assessment</h3>
            <span className="toggleIcon">{expandedSections.impact ? "▼" : "▶"}</span>
          </div>
          {expandedSections.impact && (
            <div className="cardContent">
              {Object.entries(result.affected_population).map(([attribute, impact]) => {
                const totalAffected = typeof impact.total_affected_individuals === 'number' 
                  ? impact.total_affected_individuals 
                  : parseInt(impact.total_affected_individuals) || 0;
                
                // Filter out zero-value groups and sort by count
                const filteredGroups = impact.affected_groups
                  ?.filter(g => g.disadvantaged_count > 0)
                  .sort((a, b) => b.disadvantaged_count - a.disadvantaged_count) || [];
                
                const topGroups = filteredGroups.slice(0, 3);
                const remainingGroups = filteredGroups.slice(3);
                
                return (
                  <div key={attribute} className="impactItem">
                    <div className="impactHeader">
                      <strong>{attribute}</strong>
                      <span className="impactTotal">{totalAffected.toLocaleString()} affected</span>
                    </div>
                    {topGroups.length > 0 && (
                      <div className="affectedGroups">
                        <strong>Top impacted groups:</strong>
                        {topGroups.map((group, idx) => (
                          <div key={idx} className="affectedGroup">
                            <span>{group.group}</span>
                            <span className="groupCount">{group.disadvantaged_count.toLocaleString()}</span>
                          </div>
                        ))}
                        {remainingGroups.length > 0 && (
                          <button 
                            className="viewAllButton"
                            onClick={(e) => {
                              e.stopPropagation();
                              // Toggle view all for this attribute
                              const element = e.target.closest('.impactItem');
                              element.classList.toggle('showAll');
                            }}
                          >
                            {remainingGroups.length > 0 ? `View all (${filteredGroups.length} groups)` : ''}
                          </button>
                        )}
                        {remainingGroups.length > 0 && (
                          <div className="remainingGroups">
                            {remainingGroups.map((group, idx) => (
                              <div key={idx} className="affectedGroup">
                                <span>{group.group}</span>
                                <span className="groupCount">{group.disadvantaged_count.toLocaleString()}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="insightCard animate-slideUp delay-3">
          <div className="cardHeader" onClick={() => toggleSection("recommendations")}>
            <h3>Recommendations</h3>
            <span className="toggleIcon">{expandedSections.recommendations ? "▼" : "▶"}</span>
          </div>
          {expandedSections.recommendations && (
            <div className="cardContent">
              {result.structured_bias_report && result.structured_bias_report.recommendations ? (
                <>
                  {result.structured_bias_report.recommendations.urgent_actions && result.structured_bias_report.recommendations.urgent_actions.length > 0 && (
                    <div className="recommendationSection">
                      <h4>🔴 Urgent Actions</h4>
                      {result.structured_bias_report.recommendations.urgent_actions.map((rec, idx) => (
                        <div key={idx} className="recommendationItem urgent">
                          <div className="recHeader">
                            <strong>{rec.action}</strong>
                            <span className="recTimeline">{rec.timeline}</span>
                          </div>
                          <p>{rec.reason}</p>
                          <span className="recPriority">Priority: {rec.priority}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {result.structured_bias_report.recommendations.monitor_actions && result.structured_bias_report.recommendations.monitor_actions.length > 0 && (
                    <div className="recommendationSection">
                      <h4>🟡 Monitor</h4>
                      {result.structured_bias_report.recommendations.monitor_actions.map((rec, idx) => (
                        <div key={idx} className="recommendationItem monitor">
                          <div className="recHeader">
                            <strong>{rec.action}</strong>
                            <span className="recTimeline">{rec.timeline}</span>
                          </div>
                          <p>{rec.reason}</p>
                          <span className="recPriority">Priority: {rec.priority}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {result.structured_bias_report.recommendations.safe_actions && result.structured_bias_report.recommendations.safe_actions.length > 0 && (
                    <div className="recommendationSection">
                      <h4>🟢 Safe / No Action</h4>
                      {result.structured_bias_report.recommendations.safe_actions.map((rec, idx) => (
                        <div key={idx} className="recommendationItem safe">
                          <div className="recHeader">
                            <strong>{rec.action}</strong>
                            <span className="recTimeline">{rec.timeline}</span>
                          </div>
                          <p>{rec.reason}</p>
                          <span className="recPriority">Priority: {rec.priority}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="recommendationSection">
                    <h4>Urgent Actions</h4>
                    {result.preprocessing_steps.filter(s => s.priority === "HIGH").map((step, idx) => (
                      <div key={idx} className="recommendationItem urgent">
                        <strong>{step.recommendation}</strong>
                        <p>{step.issue}</p>
                      </div>
                    ))}
                  </div>
                  <div className="recommendationSection">
                    <h4>Monitor</h4>
                    {result.preprocessing_steps.filter(s => s.priority === "MEDIUM").map((step, idx) => (
                      <div key={idx} className="recommendationItem monitor">
                        <strong>{step.recommendation}</strong>
                        <p>{step.issue}</p>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Accuracy vs Fairness Tradeoff Slider */}
        {result.tradeoff_curves && Object.keys(result.tradeoff_curves).length > 0 && (
          <div className="insightCard animate-slideUp delay-3">
            <h3>⚖️ Accuracy vs Fairness Tradeoff</h3>
            <p>Adjust the slider to simulate applying active bias mitigation. Notice how fairness improves while accuracy may slightly decrease.</p>
            
            {(() => {
              const mainCol = Object.keys(result.tradeoff_curves)[0];
              const curve = result.tradeoff_curves[mainCol];
              if (!curve) return null;
              
              const currentPoint = curve.find(c => c.mitigation_level >= tradeoffLevel) || curve[curve.length - 1];
              const basePoint = curve[0];
              const accDiff = ((currentPoint.accuracy - basePoint.accuracy) * 100).toFixed(1);
              const fairDiff = ((currentPoint.fairness - basePoint.fairness) * 100).toFixed(1);
              
              return (
                <div style={{marginTop: '20px', padding: '20px', background: '#f8f8f8', border: '1px solid #e5e5e5'}}>
                  <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    step="25" 
                    value={tradeoffLevel} 
                    onChange={e => setTradeoffLevel(parseInt(e.target.value))} 
                    style={{width: '100%', marginBottom: '20px', accentColor: '#111'}}
                  />
                  <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#666'}}>
                    <span>Base Model (0%)</span>
                    <span>Full Mitigation (100%)</span>
                  </div>
                  
                  <div style={{display: 'flex', justifyContent: 'center', marginTop: '20px', gap: '40px'}}>
                    <div style={{textAlign: 'center'}}>
                      <div style={{fontSize: '24px', fontWeight: 'bold'}}>{(currentPoint.accuracy * 100).toFixed(1)}%</div>
                      <div style={{fontSize: '14px', color: '#666'}}>Model Accuracy</div>
                      <div style={{color: accDiff < 0 ? '#EF4444' : '#666', fontSize: '12px'}}>{accDiff}% from base</div>
                    </div>
                    <div style={{textAlign: 'center'}}>
                      <div style={{fontSize: '24px', fontWeight: 'bold'}}>{(currentPoint.fairness * 100).toFixed(1)}%</div>
                      <div style={{fontSize: '14px', color: '#666'}}>Fairness (DI)</div>
                      <div style={{color: fairDiff > 0 ? '#10B981' : '#666', fontSize: '12px'}}>+{fairDiff}% from base</div>
                    </div>
                  </div>
                  
                  {tradeoffLevel > 0 && (
                    <div style={{marginTop: '20px', textAlign: 'center', fontWeight: 'bold', color: '#111'}}>
                      "Fairness +{fairDiff}% → Accuracy {accDiff}%"
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {result.simulation_result && (
          <div className="insightCard animate-slideUp delay-3">
            <div className="cardHeader" onClick={() => toggleSection("simulation")}>
              <h3>Fairness Simulation</h3>
              <span className="toggleIcon">{expandedSections.simulation ? "▼" : "▶"}</span>
            </div>
            {expandedSections.simulation && (
              <div className="cardContent">
                <div className="simulationResult">
                  <strong>Improvement: {result.simulation_result.improvement}</strong>
                  <p>DP Diff Reduced: {(result.simulation_result.dp_diff_reduced * 100).toFixed(1)}%</p>
                  <p>{result.simulation_result.explanation}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Mitigation Modal */}
      {showMitigationModal && (
        <div className="modalOverlay" onClick={() => setShowMitigationModal(false)}>
          <div className="modalContent animate-slideUp" onClick={e => e.stopPropagation()} style={{background: '#fff', padding: '30px', maxWidth: '500px', margin: '100px auto', border: '1px solid #e5e5e5'}}>
            <h2>Active Bias Mitigation</h2>
            <p style={{marginBottom: '20px', color: '#666'}}>Select a method to reduce bias in your dataset. You will need to re-upload your original CSV to apply the transformation.</p>
            
            <div className="formGroup" style={{marginBottom: '20px'}}>
              <label style={{display: 'block', fontWeight: 'bold', marginBottom: '8px'}}>Mitigation Technique</label>
              <select value={mitigationMethod} onChange={e => setMitigationMethod(e.target.value)} style={{width: '100%', padding: '10px', border: '1px solid #ccc'}}>
                <option value="reweighing">Reweighing (Adds fairness weights)</option>
                <option value="dir">Disparate Impact Remover (Transforms features)</option>
              </select>
            </div>
            
            <div className="formGroup" style={{marginBottom: '30px'}}>
              <label style={{display: 'block', fontWeight: 'bold', marginBottom: '8px'}}>Original Dataset (CSV)</label>
              <input type="file" id="mitigateFileInput" accept=".csv" style={{width: '100%'}} />
            </div>

            <div className="modalActions" style={{display: 'flex', gap: '10px', justifyContent: 'flex-end'}}>
              <button className="secondaryButton" onClick={() => setShowMitigationModal(false)}>Cancel</button>
              <button className="primaryButton" onClick={async () => {
                const fileInput = document.getElementById("mitigateFileInput");
                if (!fileInput.files[0]) {
                  alert("Please select the original CSV file.");
                  return;
                }
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                formData.append("target_column", result.detected_target);
                // Use the highest risk sensitive column
                const sensitiveCol = result.detected_sensitive_columns[0]; 
                formData.append("sensitive_column", sensitiveCol);
                formData.append("method", mitigationMethod);
                
                try {
                  const response = await fetch("http://127.0.0.1:8001/mitigate", {
                    method: "POST",
                    body: formData
                  });
                  if (!response.ok) throw new Error("Mitigation failed");
                  
                  // Trigger download
                  const blob = await response.blob();
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `mitigated_${fileInput.files[0].name}`;
                  document.body.appendChild(a);
                  a.click();
                  a.remove();
                  setShowMitigationModal(false);
                } catch(err) {
                  alert("Error: " + err.message);
                }
              }}>Apply & Download</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
