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

export async function predictFraud(transaction, threshold = 0.5) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...transaction, threshold }),
  });
  if (!res.ok) {
    let errorMessage = 'Prediction failed';
    try {
      const error = await res.json();
      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (error.detail) {
        errorMessage = JSON.stringify(error.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
    } catch (e) {
      errorMessage = `Server error: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
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
    let errorMessage = 'Receipt prediction failed';
    try {
      const error = await res.json();
      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (error.detail) {
        errorMessage = JSON.stringify(error.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
    } catch (e) {
      errorMessage = `Server error: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
  }
  return res.json();
}

export async function extractReceipt(form, file=null) {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  // Append all transaction fields as form overrides
  for (const [key, value] of Object.entries(form)) {
    if (value !== '' && value !== null && value !== undefined) {
      formData.append(key, value);
    }
  }

  const res = await fetch(`${API_BASE}/extract-receipt`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    let errorMessage = 'Receipt extraction failed';
    try {
      const error = await res.json();
      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (error.detail) {
        errorMessage = JSON.stringify(error.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
    } catch (e) {
      errorMessage = `Server error: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
  }
  return res.json();
}


export async function fileComplaint(complaintData) {
  const res = await fetch(`${API_BASE}/file-complaint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(complaintData),
  });
  if (!res.ok) {
    let errorMessage = 'Complaint filing failed';
    try {
      const error = await res.json();
      if (typeof error.detail === 'string') {
        errorMessage = error.detail;
      } else if (error.detail) {
        errorMessage = JSON.stringify(error.detail);
      } else if (error.message) {
        errorMessage = error.message;
      }
    } catch (e) {
      errorMessage = `Server error: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
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
