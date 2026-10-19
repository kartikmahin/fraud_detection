import { useState } from 'react';
import { predictFraud, predictFromReceipt, extractReceipt } from '../api';

const LOCATIONS = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune'];
const MERCHANTS = ['retail', 'online', 'grocery', 'restaurant', 'gas'];

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
  const [extracting, setExtracting] = useState(false);

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handleNumberChange = (field, raw, parser) => {
    if (raw === '' || raw === '-') {
      setForm(prev => ({ ...prev, [field]: '' }));
      return;
    }
    const parsed = parser(raw);
    setForm(prev => ({ ...prev, [field]: isNaN(parsed) ? '' : parsed }));
  };

  // Safe numeric value helper — treats empty string as 0 for display math
  const n = (v) => (v === '' || v === undefined || v === null ? 0 : Number(v));

  const stopScrollChange = (e) => e.target.blur();

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = receiptFile
        ? await predictFromReceipt(form, receiptFile, selectedModel)
        : await predictFraud(form, threshold);

      setResult({ ...res, model_used: selectedModel });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReceiptUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) {
      setReceiptFile(null);
      return;
    }

    setReceiptFile(file);
    setExtracting(true);
    setError(null);

    try {
      const result = await extractReceipt(form, file);
      
      // Update form with extracted features (final_features contains all fields with imputed defaults)
      if (result.final_features) {
        setForm(prev => ({
          ...prev,
          ...result.final_features,
        }));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setExtracting(false);
    }
  };

  const hourOfDay = Math.floor(n(form.transaction_time) / 3600);
  const minuteOfHour = Math.floor((n(form.transaction_time) % 3600) / 60);
  const timeLabel = `${String(hourOfDay).padStart(2, '0')}:${String(minuteOfHour).padStart(2, '0')}`;
  const isNight = hourOfDay >= 22 || hourOfDay < 5;

  const getRiskColor = (prob) => {
    if (prob < 0.3) return '#10b981';
    if (prob < 0.6) return '#f59e0b';
    if (prob < 0.8) return '#f97316';
    return '#ef4444';
  };

  const getRiskLevel = (prob) => {
    if (prob < 0.3) return 'LOW';
    if (prob < 0.6) return 'MEDIUM';
    if (prob < 0.8) return 'HIGH';
    return 'CRITICAL';
  };

  return (
    <div>
      <div className="page-header">
        <h2>Fraud Prediction</h2>
        <p>Analyze transactions using deep learning models for real-time fraud detection</p>
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
        {/* Transaction Form */}
        <div className="card animate-in">
          <div className="card-header">
            <span className="card-title">Transaction Details</span>
            <span className={`model-tag ${selectedModel}`}>
              {selectedModel === 'lstm' ? 'LSTM' : 'MLP'}
            </span>
          </div>

          <div className="predict-form">
            {/* Amount */}
            <div className="form-group">
              <label className="form-label">
                Transaction Amount
              </label>
              <div className="form-input-wrapper">
                <span className="form-input-prefix">INR</span>
                <input
                  type="number"
                  className="form-input"
                  value={form.transaction_amount}
                  onChange={e => handleNumberChange('transaction_amount', e.target.value, parseFloat)}
                  onWheel={stopScrollChange}
                  min="0"
                  step="0.01"
                />
              </div>
              <div className="form-hint">
                Average: INR {n(form.avg_transaction_amount)} | Ratio: {(n(form.transaction_amount) / (n(form.avg_transaction_amount) || 1)).toFixed(1)}x
              </div>
            </div>

            {/* Time of Day */}
            <div className="form-group">
              <label className="form-label">
                Transaction Time
                <span className="form-label-value">{timeLabel}{isNight ? ' (Night)' : ''}</span>
              </label>
              <input
                type="range"
                className="form-range"
                min="0"
                max="86399"
                value={form.transaction_time}
                onChange={e => handleChange('transaction_time', parseInt(e.target.value))}
              />
              <div className="form-range-labels">
                <span>00:00</span>
                <span>06:00</span>
                <span>12:00</span>
                <span>18:00</span>
                <span>23:59</span>
              </div>
              {isNight && (
                <div className="form-warning-badge">
                  Night transaction — higher risk window
                </div>
              )}
            </div>

            {/* Location & Merchant */}
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">
                  Location
                </label>
                <select
                  className="form-select"
                  value={form.location}
                  onChange={e => handleChange('location', e.target.value)}
                >
                  {LOCATIONS.map(loc => (
                    <option key={loc} value={loc}>{loc}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">
                  Merchant Category
                </label>
                <select
                  className="form-select"
                  value={form.merchant_category}
                  onChange={e => handleChange('merchant_category', e.target.value)}
                >
                  {MERCHANTS.map(m => (
                    <option key={m} value={m}>{m.charAt(0).toUpperCase() + m.slice(1)}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Account Info */}
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">
                  Account Age (days)
                </label>
                <input
                  type="number"
                  className="form-input"
                  value={form.account_age_days}
                  onChange={e => handleNumberChange('account_age_days', e.target.value, parseInt)}
                  onWheel={stopScrollChange}
                  min="0"
                />
                {n(form.account_age_days) < 90 && (
                  <div className="form-warning-badge">
                    New account — elevated risk
                  </div>
                )}
              </div>
              <div className="form-group">
                <label className="form-label">
                  Transactions (24h)
                </label>
                <input
                  type="number"
                  className="form-input"
                  value={form.transaction_count_24h}
                  onChange={e => handleNumberChange('transaction_count_24h', e.target.value, parseInt)}
                  onWheel={stopScrollChange}
                  min="0"
                />
                {n(form.transaction_count_24h) > 10 && (
                  <div className="form-warning-badge">
                    High frequency detected
                  </div>
                )}
              </div>
            </div>

            {/* Avg Transaction & Device */}
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">
                  Avg Transaction (INR)
                </label>
                <input
                  type="number"
                  className="form-input"
                  value={form.avg_transaction_amount}
                  onChange={e => handleNumberChange('avg_transaction_amount', e.target.value, parseFloat)}
                  onWheel={stopScrollChange}
                  min="0"
                  step="0.01"
                />
              </div>
              <div className="form-group">
                <label className="form-label">
                  Device ID
                </label>
                <input
                  type="text"
                  className="form-input"
                  value={form.device_id}
                  onChange={e => handleChange('device_id', e.target.value)}
                />
              </div>
            </div>

            {/* Receipt Upload */}
            <div className="form-group">
              <label className="form-label">
                Receipt Upload
                <span className="form-label-optional">Optional</span>
              </label>
              <div className="form-file-upload">
                <input
                  type="file"
                  id="receipt-upload"
                  className="form-file-input"
                  onChange={handleReceiptUpload}
                  accept="image/*,.pdf"
                  disabled={extracting}
                />
                <label htmlFor="receipt-upload" className="form-file-label">
                  {receiptFile ? (
                    <>
                      <span className="form-file-name">{receiptFile.name}</span>
                      <span
                        className="form-file-clear"
                        onClick={(e) => { e.preventDefault(); setReceiptFile(null); document.getElementById('receipt-upload').value = ''; }}
                        disabled={extracting}
                      >
                        x
                      </span>
                    </>
                  ) : extracting ? (
                    <>
                      <span className="form-file-loading">
                        <span className="spinner" style={{ width: '14px', height: '14px', borderWidth: '2px', marginRight: '6px' }}></span>
                        Extracting features from receipt...
                      </span>
                    </>
                  ) : (
                    <>
                      <span>Drop a receipt image or <strong>browse</strong></span>
                    </>
                  )}
                </label>
              </div>
              {extracting && (
                <div className="form-hint" style={{ color: 'var(--accent)' }}>
                  <span className="spinner" style={{ width: '12px', height: '12px', borderWidth: '2px', display: 'inline-block', marginRight: '6px', verticalAlign: 'middle' }}></span>
                  Running OCR & feature extraction...
                </div>
              )}
            </div>

            {/* Threshold Slider */}
            <div className="form-group">
              <label className="form-label">
                Decision Threshold
                <span className="form-label-value">{threshold.toFixed(2)}</span>
              </label>
              <input
                type="range"
                className="form-range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={e => setThreshold(parseFloat(e.target.value))}
              />
              <div className="form-range-labels">
                <span>Lenient (0.1)</span>
                <span>Balanced (0.5)</span>
                <span>Strict (0.9)</span>
              </div>
            </div>

            {/* Submit */}
            <button
              className="btn btn-primary btn-lg"
              style={{ width: '100%', marginTop: 'var(--space-md)' }}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></div>
                  Analyzing Transaction...
                </>
              ) : (
                <>Analyze Transaction</>
              )}
            </button>
          </div>
        </div>

        {/* Results Panel */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Analysis Result</span>
            {result && (
              <span className={`model-tag ${result.model_used}`}>
                {result.model_used === 'lstm' ? 'LSTM' : 'MLP'}
              </span>
            )}
          </div>

          {/* Loading State */}
          {loading && (
            <div className="loading-overlay" style={{ minHeight: '400px' }}>
              <div className="spinner"></div>
              <span>Running {selectedModel === 'lstm' ? 'LSTM sequential' : 'MLP'} inference...</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Preprocessing → Feature Engineering → Model Inference → Risk Assessment
              </span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="prediction-result danger" style={{ margin: 'var(--space-lg) 0' }}>
              <div className="result-label">Prediction Failed</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '8px' }}>{error}</p>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="predict-result-display">
              {/* Risk Gauge */}
              <div className="risk-gauge-container">
                <svg viewBox="0 0 200 120" className="risk-gauge">
                  {/* Background arc */}
                  <path
                    d="M 20 100 A 80 80 0 0 1 180 100"
                    fill="none"
                    stroke="var(--border-subtle)"
                    strokeWidth="12"
                    strokeLinecap="round"
                  />
                  {/* Filled arc */}
                  <path
                    d="M 20 100 A 80 80 0 0 1 180 100"
                    fill="none"
                    stroke={getRiskColor(result.fraud_probability)}
                    strokeWidth="12"
                    strokeLinecap="round"
                    strokeDasharray={`${result.fraud_probability * 251.3} 251.3`}
                    style={{ transition: 'stroke-dasharray 1s ease, stroke 0.5s ease' }}
                  />
                  {/* Probability text */}
                  <text x="100" y="78" textAnchor="middle" fill="var(--text-primary)" fontSize="28" fontWeight="800">
                    {(result.fraud_probability * 100).toFixed(1)}%
                  </text>
                  <text x="100" y="98" textAnchor="middle" fill="var(--text-muted)" fontSize="11">
                    Fraud Probability
                  </text>
                </svg>
              </div>

              {/* Verdict */}
              <div className={`prediction-result ${result.is_fraud ? 'danger' : 'safe'}`}>
                <div className="result-label">
                  {result.is_fraud ? 'Fraudulent Transaction' : 'Legitimate Transaction'}
                </div>
                <div className="result-message">
                  {result.message || (result.is_fraud
                    ? 'This transaction exhibits patterns consistent with fraudulent activity.'
                    : 'This transaction appears to be legitimate based on model analysis.')}
                </div>
              </div>

              {/* Risk Level Badge */}
              <div className="risk-level-badge" style={{
                '--risk-color': getRiskColor(result.fraud_probability),
              }}>
                <div className="risk-level-dot"></div>
                <span className="risk-level-text">Risk Level: {result.risk_level || getRiskLevel(result.fraud_probability)}</span>
              </div>

              {/* Detail Stats */}
              <div className="predict-detail-grid">
                <div className="predict-detail-item">
                  <div className="predict-detail-label">Model</div>
                  <div className="predict-detail-value">
                    {result.model_used === 'lstm' ? 'LSTM (BiLSTM)' : 'MLP (6-Layer DNN)'}
                  </div>
                </div>
                <div className="predict-detail-item">
                  <div className="predict-detail-label">Probability</div>
                  <div className="predict-detail-value" style={{ color: getRiskColor(result.fraud_probability) }}>
                    {(result.fraud_probability * 100).toFixed(4)}%
                  </div>
                </div>
                <div className="predict-detail-item">
                  <div className="predict-detail-label">Threshold</div>
                  <div className="predict-detail-value">{(threshold * 100).toFixed(0)}%</div>
                </div>
                <div className="predict-detail-item">
                  <div className="predict-detail-label">Verdict</div>
                  <div className="predict-detail-value" style={{ color: result.is_fraud ? 'var(--danger)' : 'var(--success)' }}>
                    {result.is_fraud ? 'BLOCK' : 'APPROVE'}
                  </div>
                </div>
              </div>

              {/* Transaction Summary */}
              <div className="predict-summary">
                <div className="predict-summary-title">Transaction Summary</div>
                <div className="predict-summary-grid">
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">Amount</span>
                    <span className="predict-summary-val">INR {form.transaction_amount.toLocaleString()}</span>
                  </div>
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">Time</span>
                    <span className="predict-summary-val">{timeLabel}</span>
                  </div>
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">Location</span>
                    <span className="predict-summary-val">{form.location}</span>
                  </div>
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">Merchant</span>
                    <span className="predict-summary-val">{form.merchant_category}</span>
                  </div>
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">Account Age</span>
                    <span className="predict-summary-val">{form.account_age_days} days</span>
                  </div>
                  <div className="predict-summary-item">
                    <span className="predict-summary-key">24h Txn Count</span>
                    <span className="predict-summary-val">{form.transaction_count_24h}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!loading && !result && !error && (
            <div className="empty-state" style={{ minHeight: '400px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <p>Enter transaction details and click "Analyze"</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px', maxWidth: '300px', margin: '8px auto 0' }}>
                The {selectedModel === 'lstm' ? 'LSTM model analyzes sequential patterns across customer transactions' : 'MLP model analyzes individual transaction features through a 6-layer deep neural network'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
