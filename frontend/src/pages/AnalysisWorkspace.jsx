import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeDataset, analyzeText } from "../api";

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
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const datasetSteps = [
    "Uploading file",
    "Detecting columns",
    "Calculating fairness metrics",
    "Analyzing bias drivers",
    "Generating recommendations",
    "Complete"
  ];

  const textSteps = [
    "Processing text",
    "Running rule-based detection",
    "Running ML classification",
    "Combining results",
    "Generating alternatives",
    "Complete"
  ];

  const handleDatasetSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose a CSV file.");
      return;
    }

    setLoading(true);
    setProcessing(true);
    setCurrentStep(0);
    setError("");
    
    try {
      // Simulate step-by-step progress
      setCurrentStep(0);
      await new Promise(r => setTimeout(r, 500));
      
      setCurrentStep(1);
      const data = await analyzeDataset({ file });
      
      setCurrentStep(2);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(3);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(4);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(5);
      setDatasetResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setProcessing(false);
      setCurrentStep(0);
    }
  };

  const handleTextSubmit = async (event) => {
    event.preventDefault();
    if (!text.trim()) {
      setError("Please enter text to analyze.");
      return;
    }

    setLoading(true);
    setProcessing(true);
    setCurrentStep(0);
    setError("");
    
    try {
      // Simulate step-by-step progress
      setCurrentStep(0);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(1);
      const data = await analyzeText({ text });
      
      setCurrentStep(2);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(3);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(4);
      await new Promise(r => setTimeout(r, 300));
      
      setCurrentStep(5);
      setTextResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setProcessing(false);
      setCurrentStep(0);
    }
  };

  const handleViewFullInsights = (type) => {
    if (type === "dataset" && datasetResult) {
      navigate("/dashboard", { state: { result: datasetResult, type: "dataset", originalFile: file } });
    } else if (type === "text" && textResult) {
      navigate("/dashboard", { state: { result: textResult, type: "text" } });
    }
  };

  return (
    <div className="workspace animate-fadeIn">
      <header className="workspaceHeader animate-slideUp">
        <div>
          <p className="eyebrow">FairSight Core</p>
          <h1>Analysis Workspace</h1>
          <p className="subtitle">Upload datasets or enter text to detect bias and unfair patterns</p>
        </div>
      </header>

      <div className="tabs">
        <button
          className={`tab ${activeTab === "dataset" ? "active" : ""}`}
          onClick={() => setActiveTab("dataset")}
        >
          Dataset Analysis
        </button>
        <button
          className={`tab ${activeTab === "text" ? "active" : ""}`}
          onClick={() => setActiveTab("text")}
        >
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
                  onChange={(event) => setFile(event.target.files?.[0] || null)}
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
                    <span className="uploadIcon">📤</span>
                    <p>Click to upload or drag and drop</p>
                    <small>CSV files only</small>
                  </div>
                )}
              </div>

              <button className="primaryButton" type="submit" disabled={loading || !file}>
                {loading ? "Analyzing..." : "Analyze Dataset"}
              </button>
            </form>

            {processing && <ProcessingIndicator steps={datasetSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}
          </div>

          {datasetResult && (
            <div className="resultPreviewCard animate-slideUp delay-2">
              <div className="resultHeader">
                <h3>Analysis Result</h3>
                <span className={`statusBadge ${datasetResult.potential_bias_detected ? "status-danger" : "status-safe"}`}>
                  {datasetResult.potential_bias_detected ? "Bias Detected" : "No Bias"}
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
                View Full Insights
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="analysisSection">
          <div className="textCard animate-slideUp delay-1">
            <h2>Text Analysis</h2>
            <p className="uploadHint">Enter text to analyze for potential bias</p>
            
            <form className="textForm" onSubmit={handleTextSubmit}>
              <textarea
                className="textInput"
                rows={6}
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Enter text to analyze for potential bias, discrimination, or unfair patterns..."
              />
              <button className="primaryButton" type="submit" disabled={loading || !text.trim()}>
                {loading ? "Analyzing..." : "Analyze Text"}
              </button>
            </form>

            {processing && <ProcessingIndicator steps={textSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}
          </div>

          {textResult && (
            <div className="resultPreviewCard animate-slideUp delay-2">
              <div className="resultHeader">
                <h3>Analysis Result</h3>
                <span className={`statusBadge ${
                  textResult.bias_detected === "Yes" ? "status-danger" :
                  textResult.bias_detected === "Possible" ? "status-warning" :
                  "status-safe"
                }`}>
                  {textResult.bias_detected === "Yes" ? "Bias Detected" :
                   textResult.bias_detected === "Possible" ? "Possible Bias" :
                   "No Bias"}
                </span>
              </div>
              <div className="resultContent">
                <div className="resultItem">
                  <span className="resultLabel">Confidence</span>
                  <span className="resultValue">{textResult.overall_confidence}</span>
                </div>
                {textResult.ml_confidence && (
                  <div className="resultItem">
                    <span className="resultLabel">ML Confidence</span>
                    <span className="resultValue">{(textResult.ml_confidence * 100).toFixed(1)}%</span>
                  </div>
                )}
                {textResult.biases.length > 0 && (
                  <div className="resultItem">
                    <span className="resultLabel">Bias Types</span>
                    <span className="resultValue">{textResult.biases.map(b => b.type).join(", ")}</span>
                  </div>
                )}
                <p className="resultSummary">{textResult.summary}</p>
              </div>
              <button className="secondaryButton" onClick={() => handleViewFullInsights("text")}>
                View Full Insights
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
