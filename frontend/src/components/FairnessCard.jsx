import React from 'react';

const getScoreColor = (score) => {
  if (score >= 80) return '#10B981';
  if (score >= 50) return '#F59E0B';
  return '#EF4444';
};

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

export default function FairnessCard({ column, metric }) {
  const fairnessScore = Math.round(metric.di_ratio * 100);
  
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

  return (
    <div className={`fairnessCard ${isHighRisk ? 'high-risk-card' : ''} ${isLowConfidence ? 'low-confidence-card' : ''}`}>
      <div className="fairnessCardHeader">
        <h3>{column}</h3>
        <span className={`riskBadge risk-${metric.severity.toLowerCase()}`}>
          {metric.severity}
        </span>
      </div>
      
      {isLowConfidence && (
        <div className="confidenceWarning">
          <span className="warningIcon"></span>
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
}
