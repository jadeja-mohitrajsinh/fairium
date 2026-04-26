import React from 'react';

export default function ExecutiveSummary({ summary, report }) {
  if (!report || !report.overall_summary) return null;
  
  const { overall_summary } = report;
  const complianceStatus = overall_summary.compliance_status || "Safe";
  
  return (
    <div className="executiveSummaryCard animate-slideUp delay-1">
      <div className="executiveHeader">
        <h3>Executive Summary</h3>
        <span className={`decisionLabel decision-${complianceStatus.toLowerCase().replace(' ', '-')}`}>
          {complianceStatus}
        </span>
      </div>
      <p className="executiveText">{overall_summary.executive_summary}</p>
      
      {overall_summary.recommended_decision && (
        <div className="decisionBox">
          <strong>Recommended Decision:</strong>
          <p>{overall_summary.recommended_decision}</p>
        </div>
      )}
      
      {overall_summary.reliability_warning && (
        <div className="reliabilityWarning">
          {overall_summary.reliability_warning}
        </div>
      )}
    </div>
  );
}
