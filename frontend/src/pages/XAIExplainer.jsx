import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { explainFeatureImportance, explainPredictions, generateCounterfactuals } from "../api";

function FeatureImportanceChart({ data }) {
  if (!data || !data.length) return null;

  const maxImportance = Math.max(...data.map(d => d.importance));

  return (
    <div className="xaiChart">
      <h4>Feature Importance</h4>
      <div className="importanceBars">
        {data.map((item, idx) => (
          <div key={idx} className="importanceBarRow">
            <span className="featureName" title={item.feature}>{item.feature}</span>
            <div className="importanceBarContainer">
              <div
                className="importanceBar"
                style={{ width: `${(item.importance / maxImportance) * 100}%` }}
              >
                <span className="importanceValue">{item.importance.toFixed(3)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GroupComparisonTable({ groups, divergentFeatures }) {
  if (!groups) return null;

  const groupNames = Object.keys(groups);
  if (groupNames.length < 2) return null;

  return (
    <div className="groupComparison">
      <h4>Per-Group Feature Importance</h4>
      <div className="comparisonGrid">
        {groupNames.map(group => (
          <div key={group} className="groupCard">
            <h5>Group: {group}</h5>
            <span className="sampleSize">n={groups[group].sample_size}</span>
            <ul className="groupFeatures">
              {groups[group].top_features?.slice(0, 5).map((f, i) => (
                <li key={i}>
                  <span className="featureRank">#{i + 1}</span>
                  <span className="featureNameSmall">{f.feature}</span>
                  <span className="featureImp">{f.importance.toFixed(3)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {divergentFeatures?.length > 0 && (
        <div className="divergentFeatures">
          <h5>⚠️ Most Divergent Features (Potential Bias Indicators)</h5>
          <ul>
            {divergentFeatures.slice(0, 5).map((item, idx) => (
              <li key={idx} className="divergentItem">
                <strong>{item.feature}</strong>
                <span className="diffBadge">
                  Diff: {item.max_difference?.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function BiasIndicators({ indicators }) {
  if (!indicators || !indicators.length) return null;

  return (
    <div className="biasIndicators">
      <h4>🚨 Potential Bias Indicators</h4>
      {indicators.map((indicator, idx) => (
        <div key={idx} className={`biasAlert bias-${indicator.severity?.toLowerCase()}`}>
          <span className="severityBadge">{indicator.severity}</span>
          <p>{indicator.explanation}</p>
          <span className="indicatorType">{indicator.indicator_type}</span>
        </div>
      ))}
    </div>
  );
}

function PredictionExplanations({ explanations }) {
  if (!explanations || !explanations.length) return null;

  return (
    <div className="predictionExplanations">
      <h4>Individual Prediction Explanations</h4>
      <div className="explanationsList">
        {explanations.map((exp, idx) => (
          <div key={idx} className="explanationCard">
            <div className="explanationHeader">
              <span className="instanceIndex">Instance #{exp.index}</span>
              <span className={`predictionBadge pred-${exp.prediction > 0.5 ? "positive" : "negative"}`}>
                Prediction: {exp.prediction > 0.5 ? "Positive" : "Negative"} ({exp.prediction.toFixed(2)})
              </span>
            </div>

            <p className="explanationSummary">{exp.explanation_summary}</p>

            <div className="featureContributions">
              <div className="positiveFeatures">
                <h6>Features that INCREASED the score:</h6>
                {exp.top_positive_features?.length ? (
                  <ul>
                    {exp.top_positive_features.map((f, i) => (
                      <li key={i}>
                        <span className="contribFeature">{f.feature}</span>
                        <span className="contribValue positive">+{f.contribution.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                ) : <span className="noneText">None</span>}
              </div>

              <div className="negativeFeatures">
                <h6>Features that DECREASED the score:</h6>
                {exp.top_negative_features?.length ? (
                  <ul>
                    {exp.top_negative_features.map((f, i) => (
                      <li key={i}>
                        <span className="contribFeature">{f.feature}</span>
                        <span className="contribValue negative">{f.contribution.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                ) : <span className="noneText">None</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CounterfactualCard({ result }) {
  if (!result || !result.counterfactuals) return null;

  return (
    <div className="counterfactualSection">
      <h4>Counterfactual Explanations</h4>

      <div className="originalInstance">
        <h5>Original Instance #{result.instance_index}</h5>
        <div className="predictionCompare">
          <span className={`origPred ${result.current_prediction ? "positive" : "negative"}`}>
            Current: {result.current_prediction ? "Positive" : "Negative"}
          </span>
          <span className="arrow">→</span>
          <span className={`targetPred ${result.desired_outcome ? "positive" : "negative"}`}>
            Target: {result.desired_outcome ? "Positive" : "Negative"}
          </span>
        </div>
      </div>

      {result.bias_analysis?.would_need_to_change_sensitive && (
        <div className="cfBiasAlert">
          <span className="alertIcon">⚠️</span>
          <strong>Bias Detected:</strong> To achieve the desired outcome,
          protected attributes would need to change:
          {result.bias_analysis.sensitive_attributes_in_changes?.map((change, idx) => (
            <span key={idx} className="sensitiveChange">
              {" "}{change.feature} from &quot;{change.original_value}&quot; to &quot;{change.change?.to}&quot;
            </span>
          ))}
        </div>
      )}

      <div className="counterfactualsList">
        <h6>Counterfactual Examples (from dataset):</h6>
        {result.counterfactuals.map((cf, idx) => (
          <div key={idx} className="counterfactualCard">
            <div className="cfHeader">
              <span className="cfNumber">Option {idx + 1}</span>
              <span className="cfDistance">Distance: {cf.distance?.toFixed(3)}</span>
              <span className="cfChanges">{cf.num_features_changed} changes</span>
            </div>
            <div className="cfChangesList">
              {Object.entries(cf.changes || {}).map(([feature, change], cidx) => (
                <div key={cidx} className="cfChangeItem">
                  <span className="changeFeature">{feature}</span>
                  <span className="changeArrow">{change.from} → {change.to}</span>
                  <span className={`changeType ${change.change_type}`}>{change.change_type}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {result.summary && (
        <div className="cfSummary">
          <p>{result.summary}</p>
        </div>
      )}
    </div>
  );
}

export default function XAIExplainer() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [predictionColumn, setPredictionColumn] = useState("");
  const [sensitiveColumn, setSensitiveColumn] = useState("");
  const [featureColumns, setFeatureColumns] = useState("");
  const [instanceIndex, setInstanceIndex] = useState(0);
  const [desiredOutcome, setDesiredOutcome] = useState(1);
  const [activeTab, setActiveTab] = useState("importance");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleFileChange = useCallback((e) => {
    setFile(e.target.files?.[0] || null);
    setError("");
    setResult(null);
  }, []);

  const handleFeatureImportance = async () => {
    if (!file) {
      setError("Please upload a CSV file first");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await explainFeatureImportance({
        file,
        predictionColumn: predictionColumn || undefined,
        sensitiveColumn: sensitiveColumn || undefined,
        featureColumns: featureColumns || undefined,
      });
      setResult({ type: "importance", data });
    } catch (err) {
      setError(err.message || "Failed to analyze feature importance");
    } finally {
      setLoading(false);
    }
  };

  const handleExplainPredictions = async () => {
    if (!file) {
      setError("Please upload a CSV file first");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await explainPredictions({
        file,
        predictionColumn: predictionColumn || undefined,
        featureColumns: featureColumns || undefined,
        numSamples: 5,
      });
      setResult({ type: "predictions", data });
    } catch (err) {
      setError(err.message || "Failed to explain predictions");
    } finally {
      setLoading(false);
    }
  };

  const handleCounterfactuals = async () => {
    if (!file) {
      setError("Please upload a CSV file first");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await generateCounterfactuals({
        file,
        predictionColumn: predictionColumn || undefined,
        featureColumns: featureColumns || undefined,
        sensitiveColumns: sensitiveColumn || undefined,
        instanceIndex,
        desiredOutcome,
        numCounterfactuals: 5,
      });
      setResult({ type: "counterfactual", data });
    } catch (err) {
      setError(err.message || "Failed to generate counterfactuals");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="xaiContainer">
      <header className="xaiHeader">
        <button onClick={() => navigate("/")} className="backButton">← Back</button>
        <h1>Explainable AI (XAI) Analysis</h1>
        <p className="subtitle">
          Understand why your model makes specific decisions and detect bias through SHAP explanations and counterfactuals.
        </p>
      </header>

      <div className="xaiContent">
        <aside className="xaiSidebar">
          <div className="uploadSection">
            <h3>Upload Dataset</h3>
            <div className="fileUpload">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                id="csv-upload"
              />
              <label htmlFor="csv-upload" className="fileLabel">
                {file ? file.name : "Choose CSV file"}
              </label>
            </div>
            {file && <p className="fileInfo">Selected: {file.name}</p>}
          </div>

          <div className="configSection">
            <h3>Configuration</h3>
            <div className="configField">
              <label>Prediction Column (optional)</label>
              <input
                type="text"
                value={predictionColumn}
                onChange={(e) => setPredictionColumn(e.target.value)}
                placeholder="e.g., prediction, y_pred"
              />
              <small>Auto-detected if not specified</small>
            </div>

            <div className="configField">
              <label>Sensitive Column (optional)</label>
              <input
                type="text"
                value={sensitiveColumn}
                onChange={(e) => setSensitiveColumn(e.target.value)}
                placeholder="e.g., gender, race"
              />
              <small>For per-group analysis</small>
            </div>

            <div className="configField">
              <label>Feature Columns (optional)</label>
              <input
                type="text"
                value={featureColumns}
                onChange={(e) => setFeatureColumns(e.target.value)}
                placeholder="comma-separated list"
              />
              <small>Auto-detected numeric columns if not specified</small>
            </div>
          </div>

          <div className="tabsSection">
            <h3>Analysis Type</h3>
            <div className="tabButtons">
              <button
                className={activeTab === "importance" ? "active" : ""}
                onClick={() => setActiveTab("importance")}
              >
                📊 Feature Importance
              </button>
              <button
                className={activeTab === "predictions" ? "active" : ""}
                onClick={() => setActiveTab("predictions")}
              >
                🔍 Explain Predictions
              </button>
              <button
                className={activeTab === "counterfactual" ? "active" : ""}
                onClick={() => setActiveTab("counterfactual")}
              >
                🔄 Counterfactuals
              </button>
            </div>
          </div>

          {activeTab === "counterfactual" && (
            <div className="cfConfig">
              <div className="configField">
                <label>Instance Index</label>
                <input
                  type="number"
                  min={0}
                  value={instanceIndex}
                  onChange={(e) => setInstanceIndex(parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="configField">
                <label>Desired Outcome</label>
                <select
                  value={desiredOutcome}
                  onChange={(e) => setDesiredOutcome(parseInt(e.target.value))}
                >
                  <option value={1}>Positive (1)</option>
                  <option value={0}>Negative (0)</option>
                </select>
              </div>
            </div>
          )}

          <button
            className="runAnalysisBtn"
            onClick={
              activeTab === "importance"
                ? handleFeatureImportance
                : activeTab === "predictions"
                ? handleExplainPredictions
                : handleCounterfactuals
            }
            disabled={loading || !file}
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>

          {error && <div className="errorBox">{error}</div>}
        </aside>

        <main className="xaiMain">
          {!result && !loading && (
            <div className="emptyState">
              <div className="emptyIcon">🔍</div>
              <h3>Ready to Explain</h3>
              <p>Upload a CSV file with model predictions and run an analysis to see SHAP explanations, feature importance by group, or counterfactual examples.</p>

              <div className="featureCards">
                <div className="featureCard">
                  <span className="icon">📊</span>
                  <h4>Feature Importance</h4>
                  <p>See which features drive predictions overall and per demographic group. Detect if different groups are evaluated differently.</p>
                </div>
                <div className="featureCard">
                  <span className="icon">🔍</span>
                  <h4>Prediction Explanations</h4>
                  <p>Understand why specific individuals received their predictions. See which features increased or decreased their score.</p>
                </div>
                <div className="featureCard">
                  <span className="icon">🔄</span>
                  <h4>Counterfactuals</h4>
                  <p>Discover what would need to change for a different outcome. If protected attributes must change, bias is indicated.</p>
                </div>
              </div>
            </div>
          )}

          {loading && (
            <div className="loadingState">
              <div className="spinner"></div>
              <p>Analyzing with SHAP and generating explanations...</p>
            </div>
          )}

          {result && result.type === "importance" && (
            <div className="resultsSection">
              <h2>Feature Importance Analysis</h2>

              {result.data.model_info && (
                <div className="modelInfo">
                  <span>Surrogate Model: {result.data.model_info.surrogate_type}</span>
                  <span>Samples: {result.data.model_info.samples}</span>
                  <span>Features: {result.data.model_info.features}</span>
                </div>
              )}

              <FeatureImportanceChart data={result.data.overall_top_features} />

              {result.data.groups && (
                <GroupComparisonTable
                  groups={result.data.groups}
                  divergentFeatures={result.data.comparative_analysis?.most_divergent_features}
                />
              )}

              <BiasIndicators indicators={result.data.potential_bias_indicators} />

              {result.data.note && <p className="note">{result.data.note}</p>}
            </div>
          )}

          {result && result.type === "predictions" && (
            <div className="resultsSection">
              <h2>Individual Prediction Explanations</h2>
              <p className="resultsMeta">
                Explained {result.data.explanations?.length} of {result.data.total_samples} instances
              </p>
              <PredictionExplanations explanations={result.data.explanations} />
            </div>
          )}

          {result && result.type === "counterfactual" && (
            <div className="resultsSection">
              <h2>Counterfactual Explanation</h2>
              <CounterfactualCard result={result.data} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
