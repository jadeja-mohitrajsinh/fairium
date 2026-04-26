export async function fetchHealth() {
  const response = await fetch("/health");
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return response.json();
}

export async function analyzeText({ text }) {
  const response = await fetch("/analyze-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    let message = `Text analysis failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function analyzeDataset({ file }) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/analyze", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let message = `Analysis failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function monitorBias({ analysisPayload, historicalRiskScores = [], thresholds = {}, scenario = "general", externalMetadata = {} }) {
  const response = await fetch("/monitor", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      analysis_payload: analysisPayload,
      historical_risk_scores: historicalRiskScores,
      thresholds,
      scenario,
      external_metadata: externalMetadata,
    }),
  });

  if (!response.ok) {
    let message = `Monitoring failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function gateDecision({
  decisionId,
  scenario = "general",
  decisionPayload,
  analysisPayload,
  riskScoreOverride,
  blockThreshold = 75,
  flagThreshold = 50,
  autoMitigation = true,
  externalMetadata = {},
}) {
  const response = await fetch("/gate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      decision_id: decisionId,
      scenario,
      decision_payload: decisionPayload,
      analysis_payload: analysisPayload,
      risk_score_override: riskScoreOverride,
      block_threshold: blockThreshold,
      flag_threshold: flagThreshold,
      auto_mitigation: autoMitigation,
      external_metadata: externalMetadata,
    }),
  });

  if (!response.ok) {
    let message = `Decision gate failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      // Keep fallback message.
    }
    throw new Error(message);
  }

  return response.json();
}
