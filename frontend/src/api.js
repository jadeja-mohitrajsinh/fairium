const API_BASE = "/api";

async function handleResponse(response) {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) message = payload.detail;
    } catch {
      // keep fallback
    }
    throw new Error(message);
  }
  return response.json();
}

export async function fetchHealth() {
  const response = await fetch("/health");
  return handleResponse(response);
}

export async function analyzeText({ text }) {
  const response = await fetch(`${API_BASE}/analyze-text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handleResponse(response);
}

export async function analyzeDataset({ file }) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/analyze`, {
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

  const response = await fetch(`${API_BASE}/mitigate`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    return handleResponse(response); // will throw
  }
  return response.blob();
}

export async function fetchSampleDatasets() {
  const response = await fetch(`${API_BASE}/datasets`);
  return handleResponse(response);
}

export async function loadSampleDataset(datasetId) {
  const response = await fetch(`${API_BASE}/datasets/${datasetId}/download`);
  if (!response.ok) {
    return handleResponse(response); // will throw
  }
  return response.blob();
}

export async function analyzeDecisions({ file, predictionColumn, actualColumn, sensitiveColumns }) {
  const formData = new FormData();
  formData.append("file", file);
  if (predictionColumn) formData.append("prediction_column", predictionColumn);
  if (actualColumn) formData.append("actual_column", actualColumn);
  if (sensitiveColumns) formData.append("sensitive_columns", sensitiveColumns);

  const response = await fetch(`${API_BASE}/analyze-decisions`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function detectColumns({ file }) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/detect-columns`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function explainFeatureImportance({ file, predictionColumn, sensitiveColumn, featureColumns }) {
  const formData = new FormData();
  formData.append("file", file);
  if (predictionColumn) formData.append("prediction_column", predictionColumn);
  if (sensitiveColumn) formData.append("sensitive_column", sensitiveColumn);
  if (featureColumns) formData.append("feature_columns", featureColumns);

  const response = await fetch(`${API_BASE}/explain/feature-importance`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function explainPredictions({ file, predictionColumn, featureColumns, sampleIndices, numSamples }) {
  const formData = new FormData();
  formData.append("file", file);
  if (predictionColumn) formData.append("prediction_column", predictionColumn);
  if (featureColumns) formData.append("feature_columns", featureColumns);
  if (sampleIndices) formData.append("sample_indices", sampleIndices);
  if (numSamples) formData.append("num_samples", numSamples.toString());

  const response = await fetch(`${API_BASE}/explain/predictions`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function generateCounterfactuals({ file, predictionColumn, featureColumns, sensitiveColumns, instanceIndex, desiredOutcome, numCounterfactuals }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("instance_index", instanceIndex.toString());
  if (predictionColumn) formData.append("prediction_column", predictionColumn);
  if (featureColumns) formData.append("feature_columns", featureColumns);
  if (sensitiveColumns) formData.append("sensitive_columns", sensitiveColumns);
  if (desiredOutcome !== undefined && desiredOutcome !== null) formData.append("desired_outcome", desiredOutcome.toString());
  if (numCounterfactuals) formData.append("num_counterfactuals", numCounterfactuals.toString());

  const response = await fetch(`${API_BASE}/explain/counterfactuals`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export async function fetchExplainInfo() {
  const response = await fetch(`${API_BASE}/explain/info`);
  return handleResponse(response);
}
