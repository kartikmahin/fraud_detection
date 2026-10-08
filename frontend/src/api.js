const API_BASE = import.meta.env.VITE_API_URL || 'https://fraud-detection-yda0.onrender.com';

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`);
  return res.json();
}

export async function getLSTMModelInfo() {
  const res = await fetch(`${API_BASE}/model-info-lstm`);
  return res.json();
}

export async function getTrainingHistory() {
  const res = await fetch(`${API_BASE}/training-history`);
  return res.json();
}

export async function getLSTMTrainingHistory() {
  const res = await fetch(`${API_BASE}/training-history-lstm`);
  return res.json();
}

export async function getMetrics() {
  const res = await fetch(`${API_BASE}/metrics`);
  return res.json();
}

export async function getLSTMMetrics() {
  const res = await fetch(`${API_BASE}/metrics-lstm`);
  return res.json();
}

export async function predictFraud(transaction) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(transaction),
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Prediction failed');
  }
  return res.json();
}

export async function predictFromReceipt(form, file=null, model='mlp') {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  // Append all transaction fields
  for (const [key, value] of Object.entries(form)) {
    formData.append(key, value);
  }
  formData.append('model', model);

  const res = await fetch(`${API_BASE}/predict-receipt`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Receipt prediction failed');
  }
  return res.json();
}


export async function trainModel() {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'Training failed');
  }
  return res.json();
}

export async function trainLSTMModel() {
  const res = await fetch(`${API_BASE}/train-lstm`, {
    method: 'POST',
  });
  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || 'LSTM Training failed');
  }
  return res.json();
}

