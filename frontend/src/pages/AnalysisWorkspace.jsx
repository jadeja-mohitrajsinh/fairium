import { useState } from "react";
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
              {index < currentStep ? "Done" : index + 1}
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
  const navigate = useNavigate();

  const datasetSteps = ["Uploading", "Analyzing", "Generating Results"];
  const textSteps = ["Processing", "Analyzing", "Generating Results"];

  const handleDatasetSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please choose a CSV file.");
      return;
    }

    setLoading(true);
    setCurrentStep(0);
    setError("");
    
    try {
      setCurrentStep(0); // Uploading
      const data = await analyzeDataset({ file });
      
      setCurrentStep(1); // Analyzing
      await new Promise(r => setTimeout(r, 200)); // Minor delay for UX
      
      setCurrentStep(2); // Generating
      setDatasetResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
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
      setCurrentStep(0); // Processing
      const data = await analyzeText({ text });
      
      setCurrentStep(1); // Analyzing
      await new Promise(r => setTimeout(r, 200));
      
      setCurrentStep(2); // Generating
      setTextResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleViewFullInsights = (type) => {
    const res = type === "dataset" ? datasetResult : textResult;
    if (res) {
      navigate("/dashboard", { state: { result: res, type } });
    }
  };

  return (
    <div className="workspace animate-fadeIn">
      <header className="workspaceHeader animate-slideUp">
        <div>
          <p className="eyebrow">FairSight Core</p>
          <h1>Analysis Workspace</h1>
          <p className="subtitle">Enterprise AI fairness analysis platform</p>
        </div>
      </header>

      <div className="tabs">
        <button className={`tab ${activeTab === "dataset" ? "active" : ""}`} onClick={() => setActiveTab("dataset")}>Dataset Analysis</button>
        <button className={`tab ${activeTab === "text" ? "active" : ""}`} onClick={() => setActiveTab("text")}>Text Analysis</button>
      </div>

      <div className="analysisSection">
        {activeTab === "dataset" ? (
          <div className="uploadCard animate-slideUp delay-1">
            <h2>Upload Dataset</h2>
            <form className="uploadForm" onSubmit={handleDatasetSubmit}>
              <div className={`dropZone ${file ? "hasFile" : ""}`} onClick={() => document.getElementById("fileInput").click()}>
                <input id="fileInput" type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} style={{ display: "none" }} />
                {file ? (
                  <div className="filePreview">
                    <span className="fileName">{file.name}</span>
                  </div>
                ) : (
                  <p>Click to upload or drag and drop CSV</p>
                )}
              </div>
              <button className="primaryButton" type="submit" disabled={loading || !file}>
                {loading ? "Analyzing..." : "Analyze Dataset"}
              </button>
            </form>
            {loading && <ProcessingIndicator steps={datasetSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}
          </div>
        ) : (
          <div className="textCard animate-slideUp delay-1">
            <h2>Text Analysis</h2>
            <form className="textForm" onSubmit={handleTextSubmit}>
              <textarea className="textInput" rows={6} value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter text..." />
              <button className="primaryButton" type="submit" disabled={loading || !text.trim()}>
                {loading ? "Analyzing..." : "Analyze Text"}
              </button>
            </form>
            {loading && <ProcessingIndicator steps={textSteps} currentStep={currentStep} />}
            {error && <div className="errorBox">{error}</div>}
          </div>
        )}

        {(activeTab === "dataset" ? datasetResult : textResult) && (
          <div className="resultPreviewCard animate-slideUp delay-2">
            <h3>Analysis Result</h3>
            <div className="resultContent">
               <p>{(activeTab === "dataset" ? datasetResult : textResult).summary}</p>
            </div>
            <button className="secondaryButton" onClick={() => handleViewFullInsights(activeTab)}>
              View Full Insights
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
