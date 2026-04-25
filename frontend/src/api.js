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
