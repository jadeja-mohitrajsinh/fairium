const API_BASE_URL = "http://localhost:8001/api";

async function handleResponse(response) {
  if (!response.ok) {
    let message = `API request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = payload.detail;
      }
    } catch {
      // Use fallback message
    }
    throw new Error(message);
  }
  return response.json();
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL.replace('/api', '')}/health`);
  return handleResponse(response);
}

export async function analyzeText({ text }) {
  const response = await fetch(`${API_BASE_URL}/analyze-text`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
  return handleResponse(response);
}

export async function analyzeDataset({ file }) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function mitigateDataset({ file, targetColumn, sensitiveColumn, method }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target_column", targetColumn);
  formData.append("sensitive_column", sensitiveColumn);
  formData.append("method", method);

  const response = await fetch(`${API_BASE_URL}/mitigate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    return handleResponse(response);
  }

  return response.blob();
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
