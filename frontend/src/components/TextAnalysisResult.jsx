import React from 'react';

export default function TextAnalysisResult({ result }) {
  if (!result) return null;

  return (
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
  );
}
