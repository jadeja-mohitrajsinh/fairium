import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeDataset, analyzeText, fetchSampleDatasets, loadSampleDataset } from "../api";
const DOMAIN_LABELS = {
  income: "💰 Income",
  hiring: "💼 Hiring",
  lending: "🏦 Lending",
  criminal_justice: "⚖️ Justice",
};

function ProcessingIndicator({ steps, currentStep }) {
  return (
    <div className="processingIndicator">
      <div className="processingHeader">
        <div className="spinner"></div>
        <span className="processingTitle">Processing...</span>
      </div>
      <div className="processingSteps">
        {steps.map((step, index) => (
          <div
            key={index}
            className={`processingStep ${index <= currentStep ? "active" : ""} ${index === currentStep ? "current" : ""}`}
          >
            <div className="stepIndicator">
              {index < currentStep ? "✓" : index + 1}
            </div>
            <span className="stepLabel">{step}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AnalysisWorkspace() {
  const [activeTab, setActiveTab] = useState("dataset");
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [datasetResult, setDatasetResult] = useState(null);
  const [textResult, setTextResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState("");
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [loadingSample, setLoadingSample] = useState(null);
  const navigate = useNavigate();

  const datasetSteps = [
    "Uploading file",
    "Detecting columns",
    "Computing fairness metrics",
    "Analyzing bias drivers",
    "Generating report",
  ];
  const textSteps = [
    "Processing text",
    "Rule-based detection",
    "ML classification",
    "Generating insights",
  ];

  useEffect(() => {
    fetchSampleDatasets()
      .then((data) => setSampleDatasets(data.datasets || []))
      .catch(() => {}); // silently fail — sample datasets are optional
  }, []);

  const runDatasetAnalysis = async (fileToAnalyze) => {
    setLoading(true);
    setCurrentStep(0);
    setError("");
    try {
      setCurrentStep(1);
      const data = await analyzeDataset({ file: fileToAnalyze });
      setCurrentStep(2);
      await new Promise((r) => setTimeout(r, 150));
      setCurrentStep(3);
      await new Promise((r) => setTimeout(r, 150));
      setCurrentStep(4);
      setDatasetResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDatasetSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose a CSV file.");
      return;
    }
    await runDatasetAnalysis(file);
  };

  const handleSampleLoad = async (datasetId) => {
    setLoadingSample(datasetId);
    setError("");
    try {
      const blob = await loadSampleDataset(datasetId);
      const ds = sampleDatasets.find((d) => d.id === datasetId);
      const sampleFile = new File([blob], `${ds?.name || datasetId}.csv`, { type: "text/csv" });
      setFile(sampleFile);
      await runDatasetAnalysis(sampleFile);
    } catch (err) {
      setError(`Failed to load sample dataset: ${err.message}`);
    } finally {
      setLoadingSample(null);
    }
  };

  const handleTextSubmit = async (event) => {
    event.preventDefault();
    if (!text.trim()) {
      setError("Please enter text to analyze.");
      return;
    }
    setLoading(true);
    setCurrentStep(0);
    setError("");
    try {
      setCurrentStep(1);
      const data = await analyzeText({ text });
      setCurrentStep(2);
      await new Promise((r) => setTimeout(r, 150));
      setCurrentStep(3);
      setTextResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleViewFullInsights = (type) => {
    const res = type === "dataset" ? datasetResult : textResult;
    if (res) navigate("/dashboard", { state: { result: res, type } });
  };

  return (
    <div className="workspace animate-fadeIn">
      <header className="workspaceHeader animate-slideUp">
        <div>
          <p className="eyebrow">FairSight Core</p>
          <h1>Analysis Workspace</h1>
          <p className="subtitle">Upload datasets or enter text to detect bias and unfair patterns</p>
        </div>
        <button className="decisionAnalysisBtn" onClick={() => navigate("/decisions")}>
          🤖 Analyze AI Decisions
        </button>
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === "dataset" ? "active" : ""}`} onClick={() => setActiveTab("dataset")}>
          Dataset Analysis
        </button>
        <button className={`tab ${activeTab === "text" ? "active" : ""}`} onClick={() => setActiveTab("text")}>
          Text Analysis
        </button>
      </div>

      {activeTab === "dataset" ? (
        <div className="analysisSection">
          <div className="uploadCard animate-slideUp delay-1">
            <h2>Upload Dataset</h2>
            <p className="uploadHint">Drag and drop your CSV file or click to browse</p>

            <form className="uploadForm" onSubmit={handleDatasetSubmit}>
              <div
                className={`dropZone ${file ? "hasFile" : ""}`}
                onClick={() => document.getElementById("fileInput").click()}
              >
                <input
                  id="fileInput"
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  style={{ display: "none" }}
                />
                {file ? (
                  <div className="filePreview">
                    <span className="fileIcon">📄</span>
                    <span className="fileName">{file.name}</span>
                    <span className="fileSize">{(file.size / 1024).toFixed(1)} KB</span>
                  </div>
                ) : (
                  <div className="dropZoneContent">
                    <span className="uploadIcon">⬆</span>
                    <p>Click to upload or drag and drop</p>
                    <small>CSV files only · max 50 MB</small>
                  </div>
                )}
              </div>

              <button className="primaryButton" type="submit" disabled={loading || !file}>
                {loading ? "Analyzing..." : "Analyze Dataset"}
              </button>
            </form>

            {loading && <ProcessingIndicator steps={datasetSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}
          </div>

          {/* Sample Datasets */}
          {sampleDatasets.length > 0 && !loading && (
            <div className="sampleDatasetsCard animate-slideUp delay-2">
              <h3>Try a Sample Dataset</h3>
              <p className="uploadHint">Real-world fairness benchmarks — click to load and analyze instantly</p>
              <div className="sampleGrid">
                {sampleDatasets.map((ds) => (
                  <button
                    key={ds.id}
                    className="sampleDatasetBtn"
                    onClick={() => handleSampleLoad(ds.id)}
                    disabled={loadingSample !== null || loading}
                  >
                    <span className="sampleDomain">{DOMAIN_LABELS[ds.domain] || ds.domain}</span>
                    <strong className="sampleName">{ds.name}</strong>
                    <span className="sampleDesc">{ds.description}</span>
                    <span className="sampleRows">{ds.rows.toLocaleString()} rows</span>
                    {loadingSample === ds.id && <span className="sampleLoading">Loading...</span>}
                  </button>
                ))}
              </div>
            </div>
          )}

          {datasetResult && (
            <div className="resultPreviewCard animate-slideUp delay-2">
              <div className="resultHeader">
                <h3>Analysis Complete</h3>
                <span className={`statusBadge ${datasetResult.potential_bias_detected ? "status-danger" : "status-safe"}`}>
                  {datasetResult.potential_bias_detected ? "⚠ Bias Detected" : "✓ No Bias"}
                </span>
              </div>
              <div className="resultContent">
                <div className="resultItem">
                  <span className="resultLabel">Target Column</span>
                  <span className="resultValue">{datasetResult.detected_target}</span>
                </div>
                <div className="resultItem">
                  <span className="resultLabel">Sensitive Attributes</span>
                  <span className="resultValue">{datasetResult.detected_sensitive_columns.join(", ")}</span>
                </div>
                <div className="resultItem">
                  <span className="resultLabel">Risk Level</span>
                  <span className={`resultValue risk-${datasetResult.bias_report_summary.overall_risk_level.toLowerCase()}`}>
                    {datasetResult.bias_report_summary.overall_risk_level}
                  </span>
                </div>
                <p className="resultSummary">{datasetResult.summary}</p>
              </div>
              <button className="secondaryButton" onClick={() => handleViewFullInsights("dataset")}>
                View Full Insights →
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="analysisSection">
          <div className="textCard animate-slideUp delay-1">
            <h2>Text Bias Analysis</h2>
            <p className="uploadHint">Detect bias, microaggressions, and discriminatory language in text</p>

            <form className="textForm" onSubmit={handleTextSubmit}>
              <textarea
                className="textInput"
                rows={6}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a job description, policy document, or any text to analyze for bias..."
              />
              <button className="primaryButton" type="submit" disabled={loading || !text.trim()}>
                {loading ? "Analyzing..." : "Analyze Text"}
              </button>
            </form>

            {loading && <ProcessingIndicator steps={textSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}

            {/* Example prompts */}
            {!text && !loading && (
              <div className="examplePrompts">
                <p className="uploadHint">Try an example:</p>
                <div className="exampleGrid">
                  {[
                    "We are looking for a young, energetic candidate who fits our traditional work culture.",
                    "Only male candidates with strong technical backgrounds should apply.",
                    "We prefer candidates from urban backgrounds with top-tier university degrees.",
                  ].map((example, i) => (
                    <button key={i} className="exampleBtn" onClick={() => setText(example)}>
                      {example.substring(0, 60)}...
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {textResult && (
            <div className="resultPreviewCard animate-slideUp delay-2">
              <div className="resultHeader">
                <h3>Analysis Result</h3>
                <span className={`statusBadge ${
                  textResult.bias_detected === "Yes" ? "status-danger" :
                  textResult.bias_detected === "Possible" ? "status-warning" : "status-safe"
                }`}>
                  {textResult.bias_detected === "Yes" ? "⚠ Bias Detected" :
                   textResult.bias_detected === "Possible" ? "~ Possible Bias" : "✓ No Bias"}
                </span>
              </div>
              <div className="resultContent">
                <div className="resultItem">
                  <span className="resultLabel">Confidence</span>
                  <span className="resultValue">{textResult.overall_confidence}</span>
                </div>
                {textResult.biases.length > 0 && (
                  <div className="resultItem">
                    <span className="resultLabel">Bias Types Found</span>
                    <span className="resultValue">{textResult.biases.map((b) => b.type).join(", ")}</span>
                  </div>
                )}
                <p className="resultSummary">{textResult.summary}</p>
              </div>
              <button className="secondaryButton" onClick={() => handleViewFullInsights("text")}>
                View Full Insights →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
