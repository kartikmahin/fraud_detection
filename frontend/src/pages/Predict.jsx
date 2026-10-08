import { useState } from 'react';
import { predictFraud, predictFraudLSTM, predictFromReceipt } from '../api';

/**
 * Generate a human-readable analysis summary based on the transaction inputs
 * and the prediction result.
 */
function generateAnalysisSummary(form, result) {
  const factors = [];
  const positiveFactors = [];

  const amount = form.transaction_amount;
  const avgAmount = form.avg_transaction_amount;
  const hourOfDay = Math.floor(form.transaction_time / 3600);
  const txCount24h = form.transaction_count_24h;
  const accountAge = form.account_age_days;
  const category = form.merchant_category;
  const amountRatio = amount / (avgAmount || 1);

  // Amount analysis
  if (amount > 2000) {
    factors.push({
      label: 'Very high transaction amount',
      detail: `₹${amount.toLocaleString()} is significantly above normal spending patterns`,
      severity: 'high',
    });
  } else if (amount > 500) {
    factors.push({
      label: 'Above-average transaction amount',
      detail: `₹${amount.toLocaleString()} is higher than typical transactions`,
      severity: 'medium',
    });
  } else {
    positiveFactors.push({
      label: 'Normal transaction amount',
      detail: `₹${amount.toLocaleString()} is within typical spending range`,
    });
  }

  // Amount vs average ratio
  if (amountRatio > 5) {
    factors.push({
      label: 'Amount far exceeds personal average',
      detail: `${amountRatio.toFixed(1)}x higher than average (₹${avgAmount})`,
      severity: 'high',
    });
  } else if (amountRatio > 2) {
    factors.push({
      label: 'Amount above personal average',
      detail: `${amountRatio.toFixed(1)}x higher than average (₹${avgAmount})`,
      severity: 'medium',
    });
  } else {
    positiveFactors.push({
      label: 'Amount consistent with spending history',
      detail: `Within ${amountRatio.toFixed(1)}x of average (₹${avgAmount})`,
    });
  }

  // Time analysis
  if (hourOfDay >= 22 || hourOfDay <= 5) {
    factors.push({
      label: 'Late night / early morning transaction',
      detail: `Transaction at ${hourOfDay}:00 — unusual hours with higher fraud rates`,
      severity: 'medium',
    });
  } else if (hourOfDay >= 9 && hourOfDay <= 20) {
    positiveFactors.push({
      label: 'Normal business hours',
      detail: `Transaction at ${hourOfDay}:00 — typical activity window`,
    });
  }

  // Transaction frequency
  if (txCount24h > 15) {
    factors.push({
      label: 'Extremely high transaction frequency',
      detail: `${txCount24h} transactions in 24h suggests potential card testing or rapid fraud`,
      severity: 'high',
    });
  } else if (txCount24h > 10) {
    factors.push({
      label: 'High transaction frequency',
      detail: `${txCount24h} transactions in 24h is above normal patterns`,
      severity: 'medium',
    });
  } else {
    positiveFactors.push({
      label: 'Normal transaction frequency',
      detail: `${txCount24h} transactions in 24h is within normal range`,
    });
  }

  // Account age
  if (accountAge < 30) {
    factors.push({
      label: 'Very new account',
      detail: `Account is only ${accountAge} days old — new accounts have higher fraud risk`,
      severity: 'high',
    });
  } else if (accountAge < 90) {
    factors.push({
      label: 'Relatively new account',
      detail: `Account is ${accountAge} days old — still within the high-risk period`,
      severity: 'medium',
    });
  } else {
    positiveFactors.push({
      label: 'Established account',
      detail: `Account is ${accountAge} days old with established history`,
    });
  }

  // Merchant category
  if (category === 'online') {
    factors.push({
      label: 'Online merchant category',
      detail: 'Online transactions carry higher fraud risk due to card-not-present nature',
      severity: 'medium',
    });
  } else {
    positiveFactors.push({
      label: `In-person ${category} merchant`,
      detail: 'Physical merchant transactions have lower fraud rates',
    });
  }

  return { factors, positiveFactors };
}

export default function Predict() {
  const [form, setForm] = useState({
    transaction_amount: 250,
    transaction_time: 43200,
    location: 'Mumbai',
    device_id: 'DEV001',
    merchant_category: 'retail',
    account_age_days: 365,
    transaction_count_24h: 3,
    avg_transaction_amount: 150,
  });
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [receiptFile, setReceiptFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('mlp');

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setAnalysisForm(null);
    try {
      let res;
      if (receiptFile) {
        // Use the receipt endpoint (file + form data)
        res = await predictFromReceipt(form, receiptFile, selectedModel);
      } else {
        const predictFn = selectedModel === 'lstm' ? predictFraudLSTM : predictFraud;
        res = await predictFn(form);
      }
      setResult({ ...res, model_used: selectedModel });
      setAnalysisForm({ ...form });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskClass = result
    ? result.risk_level === 'HIGH'
      ? 'danger'
      : result.risk_level === 'MEDIUM'
      ? 'warning'
      : 'safe'
    : '';

  const hourOfDay = Math.floor(form.transaction_time / 3600);

  const analysis = result && analysisForm ? generateAnalysisSummary(analysisForm, result) : null;

  return (
    <div>
      <div className="page-header">
        <h2>Fraud Prediction</h2>
        <p>Analyze a transaction for fraud risk using deep learning models</p>
      </div>

      {/* Model Selector */}
      <div className="model-selector">
        <span className="model-selector-label">Model</span>
        <div className="model-selector-pills">
          <button
            className={`model-pill ${selectedModel === 'mlp' ? 'active' : ''}`}
            onClick={() => setSelectedModel('mlp')}
          >
            MLP
            <span className="model-pill-badge">6-Layer DNN</span>
          </button>
          <button
            className={`model-pill ${selectedModel === 'lstm' ? 'active' : ''}`}
            onClick={() => setSelectedModel('lstm')}
          >
            LSTM
            <span className="model-pill-badge">Sequence</span>
          </button>
        </div>
      </div>

      <div className="grid-2">
        {/* Input Form */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Transaction Details</span>
            <span className={`model-tag ${selectedModel}`}>
              {selectedModel === 'lstm' ? 'LSTM Model' : 'MLP Model'}
            </span>
          </div>

          <div className="grid-2" style={{ gap: 'var(--space-md)' }}>
            <div className="form-group">
              <label className="form-label">Amount (₹)</label>
              <input
                type="number"
                className="form-input"
                value={form.transaction_amount}
                onChange={(e) => handleChange('transaction_amount', parseFloat(e.target.value) || 0)}
                min="0.01"
                step="10"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Time of Day (Hour: {hourOfDay}:00)</label>
              <input
                type="range"
                min="0"
                max="86399"
                value={form.transaction_time}
                onChange={(e) => handleChange('transaction_time', parseInt(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Location</label>
              <select
                className="form-select"
                value={form.location}
                onChange={(e) => handleChange('location', e.target.value)}
              >
                {['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad'].map((loc) => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Merchant Category</label>
              <select
                className="form-select"
                value={form.merchant_category}
                onChange={(e) => handleChange('merchant_category', e.target.value)}
              >
                {['retail', 'grocery', 'restaurant', 'gas', 'online'].map((cat) => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Account Age (days)</label>
              <input
                type="number"
                className="form-input"
                value={form.account_age_days}
                onChange={(e) => handleChange('account_age_days', parseInt(e.target.value) || 0)}
                min="0"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Transactions (24h)</label>
              <input
                type="number"
                className="form-input"
                value={form.transaction_count_24h}
                onChange={(e) => handleChange('transaction_count_24h', parseInt(e.target.value) || 0)}
                min="0"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Avg Transaction (₹)</label>
              <input
                type="number"
                className="form-input"
                value={form.avg_transaction_amount}
                onChange={(e) => handleChange('avg_transaction_amount', parseFloat(e.target.value) || 0)}
                min="0.01"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Upload Receipt (optional)</label>
              <input
                type="file"
                accept="image/*,application/pdf"
                className="form-input"
                onChange={e => setReceiptFile(e.target.files[0])}
              />
            </div>


          <div className="slider-container" style={{ marginTop: 'var(--space-md)' }}>
            <div className="slider-header">
              <label className="form-label" style={{ margin: 0 }}>Detection Threshold</label>
              <span className="slider-value">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              <span>More Alerts</span>
              <span>Fewer Alerts</span>
            </div>


          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%', marginTop: 'var(--space-md)' }}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></div>
                Analyzing with {selectedModel === 'lstm' ? 'LSTM' : 'MLP'}...
              </>
            ) : (
              <>Analyze Transaction ({selectedModel === 'lstm' ? 'LSTM' : 'MLP'})</>
            )}
          </button>
        </div>
        </div>
        
        {/* Result */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Analysis Result</span>
            {result && (
              <span className={`model-tag ${result.model_used}`}>
                {result.model_used === 'lstm' ? 'LSTM' : 'MLP'}
              </span>
            )}
          </div>

          {error && (
            <div className="prediction-result danger">
              <div className="result-label">Error</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{error}</p>
            </div>
          )}

          {result && (
            <div className={`prediction-result ${riskClass}`}>
              <div className="result-label">
                {result.is_fraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
              </div>
              <div className="result-probability" style={{ color: result.is_fraud ? 'var(--danger)' : 'var(--success)' }}>
                {(result.fraud_probability * 100).toFixed(1)}%
              </div>
              <span className={`risk-badge ${result.risk_level.toLowerCase()}`}>
                {result.risk_level} RISK
              </span>

              <div className="result-details">
                <div className="result-detail">
                  <div className="result-detail-label">Probability</div>
                  <div className="result-detail-value">{(result.fraud_probability * 100).toFixed(2)}%</div>
                </div>
                <div className="result-detail">
                  <div className="result-detail-label">Risk Level</div>
                  <div className="result-detail-value">{result.risk_level}</div>
                </div>
                <div className="result-detail">
                  <div className="result-detail-label">Model</div>
                  <div className="result-detail-value">{result.model_used === 'lstm' ? 'LSTM' : 'MLP (DNN)'}</div>
                </div>
              </div>

              {/* Probability gauge */}
              <div style={{ marginTop: 'var(--space-xl)' }}>
                <div className="progress-bar-container" style={{ height: '12px' }}>
                  <div
                    className="progress-bar-fill"
                    style={{
                      width: `${result.fraud_probability * 100}%`,
                      background: result.is_fraud
                        ? 'var(--gradient-danger)'
                        : 'var(--gradient-success)',
                    }}
                  ></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                  <span>Safe</span>
                  <span>Fraudulent</span>
                </div>
              </div>

              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: 'var(--space-lg)' }}>
                {result.message}
              </p>
            </div>
          )}

          {!result && !error && (
            <div className="empty-state">
              <p>Submit a transaction to see the analysis result</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Choose MLP or LSTM model above, then analyze the transaction
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Analysis Summary - shows after prediction */}
      {result && analysis && (
        <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
          <div className="card-header">
            <span className="card-title">
              Analysis Summary — Why this transaction is {result.is_fraud ? 'Fraudulent' : 'Legitimate'}
            </span>
          </div>

          <div className="analysis-summary-text">
            <p>
              {result.is_fraud ? (
                <>
                  This transaction has been flagged as <strong>potentially fraudulent</strong> with a {(result.fraud_probability * 100).toFixed(1)}% confidence score
                  using the <strong>{result.model_used === 'lstm' ? 'LSTM' : 'MLP'}</strong> model.
                  The following risk factors contributed to this assessment:
                </>
              ) : (
                <>
                  This transaction appears <strong>legitimate</strong> with only a {(result.fraud_probability * 100).toFixed(1)}% fraud probability
                  using the <strong>{result.model_used === 'lstm' ? 'LSTM' : 'MLP'}</strong> model.
                  The following factors support this assessment:
                </>
              )}
            </p>
          </div>

          {/* Risk Factors */}
          {analysis.factors.length > 0 && (
            <div className="analysis-section">
              <div className="analysis-section-header risk">
                <span>Risk Factors ({analysis.factors.length})</span>
              </div>
              <div className="analysis-factors-list">
                {analysis.factors.map((f, i) => (
                  <div key={i} className={`analysis-factor-item ${f.severity}`}>
                    <div className="analysis-factor-content">
                      <div className="analysis-factor-label">{f.label}</div>
                      <div className="analysis-factor-detail">{f.detail}</div>
                    </div>
                    <span className={`analysis-severity-badge ${f.severity}`}>
                      {f.severity}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Positive Factors */}
          {analysis.positiveFactors.length > 0 && (
            <div className="analysis-section">
              <div className="analysis-section-header safe">
                <span>Positive Indicators ({analysis.positiveFactors.length})</span>
              </div>
              <div className="analysis-factors-list">
                {analysis.positiveFactors.map((f, i) => (
                  <div key={i} className="analysis-factor-item positive">
                    <div className="analysis-factor-content">
                      <div className="analysis-factor-label">{f.label}</div>
                      <div className="analysis-factor-detail">{f.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Verdict summary */}
          <div className={`analysis-verdict ${result.is_fraud ? 'danger' : 'safe'}`}>
            <div className="analysis-verdict-text">
              <strong>Verdict:</strong>{' '}
              {result.is_fraud ? (
                <>
                  {analysis.factors.length} risk factor{analysis.factors.length !== 1 ? 's' : ''} detected.
                  The combination of {analysis.factors.slice(0, 2).map(f => f.label.toLowerCase()).join(' and ')} strongly
                  suggests this transaction should be reviewed or blocked.
                </>
              ) : (
                <>
                  {analysis.positiveFactors.length} positive indicator{analysis.positiveFactors.length !== 1 ? 's' : ''} found
                  {analysis.factors.length > 0 ? ` with only ${analysis.factors.length} minor risk factor${analysis.factors.length !== 1 ? 's' : ''}` : ''}.
                  This transaction follows normal behavior patterns and can be approved.
                </>
              )}
            </div>

        </div>
      )}

      {/* Quick Test Scenarios */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">Quick Test Scenarios</span>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 50,
                transaction_time: 43200,
                location: 'Mumbai',
                device_id: 'DEV001',
                merchant_category: 'grocery',
                account_age_days: 730,
                transaction_count_24h: 2,
                avg_transaction_amount: 75,
              });
              setResult(null);
              setAnalysisForm(null);
            }}
          >
            Normal Transaction
          </button>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 2500,
                transaction_time: 10800,
                location: 'Delhi',
                device_id: 'DEV099',
                merchant_category: 'online',
                account_age_days: 15,
                transaction_count_24h: 25,
                avg_transaction_amount: 50,
              });
              setResult(null);
              setAnalysisForm(null);
            }}
          >
            Suspicious Transaction
          </button>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 800,
                transaction_time: 82800,
                location: 'Bengaluru',
                device_id: 'DEV050',
                merchant_category: 'online',
                account_age_days: 60,
                transaction_count_24h: 12,
                avg_transaction_amount: 100,
              });
              setResult(null);
              setAnalysisForm(null);
            }}
          >
            Night Transaction
          </button>
        </div>
      </div>
    </div>
  );
}
