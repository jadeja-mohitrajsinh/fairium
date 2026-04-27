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
