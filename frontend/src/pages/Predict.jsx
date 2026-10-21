import { useState } from 'react';
import { predictFraud, predictFromReceipt, extractReceipt, analyzeFraud } from '../api';

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

/* Category label + color mapping for rule signals */
const CATEGORY_META = {
  ip_anomaly: { label: 'IP Anomaly', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' },
  small_amount: { label: 'Small-Amount Pattern', color: '#ec4899', bg: 'rgba(236,72,153,0.12)' },
  velocity: { label: 'Velocity', color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)' },
};

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
    // IP metadata
    ip_country: '',
    ip_is_vpn: false,
    ip_is_proxy: false,
    ip_login_count: '',
    ip_geo_distance_km: '',
    // Small-amount context
    small_txn_count_6h: '',
    cumulative_amount_6h: '',
    same_merchant_count_1h: '',
  });
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [receiptFile, setReceiptFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('mlp');
  const [analysisForm, setAnalysisForm] = useState(null);
  const [extracting, setExtracting] = useState(false);
  const [analysisMode, setAnalysisMode] = useState('full'); // 'full' (ML+rules) or 'ml_only'
  const [showIpFields, setShowIpFields] = useState(false);
  const [showPatternFields, setShowPatternFields] = useState(false);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  /* Build the payload — strip empty optional fields */
  const buildPayload = () => {
    const payload = {
      transaction_amount: form.transaction_amount,
      transaction_time: form.transaction_time,
      location: form.location,
      device_id: form.device_id,
      merchant_category: form.merchant_category,
      account_age_days: form.account_age_days,
      transaction_count_24h: form.transaction_count_24h,
      avg_transaction_amount: form.avg_transaction_amount,
    };
    // IP fields — only include if user provided them
    if (form.ip_country) payload.ip_country = form.ip_country;
    if (form.ip_is_vpn) payload.ip_is_vpn = true;
    if (form.ip_is_proxy) payload.ip_is_proxy = true;
    if (form.ip_login_count !== '' && form.ip_login_count !== null)
      payload.ip_login_count = parseInt(form.ip_login_count) || 0;
    if (form.ip_geo_distance_km !== '' && form.ip_geo_distance_km !== null)
      payload.ip_geo_distance_km = parseFloat(form.ip_geo_distance_km) || 0;
    // Small-amount pattern fields
    if (form.small_txn_count_6h !== '' && form.small_txn_count_6h !== null)
      payload.small_txn_count_6h = parseInt(form.small_txn_count_6h) || 0;
    if (form.cumulative_amount_6h !== '' && form.cumulative_amount_6h !== null)
      payload.cumulative_amount_6h = parseFloat(form.cumulative_amount_6h) || 0;
    if (form.same_merchant_count_1h !== '' && form.same_merchant_count_1h !== null)
      payload.same_merchant_count_1h = parseInt(form.same_merchant_count_1h) || 0;
    return payload;
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
        res._mode = 'ml_only';
      } else if (analysisMode === 'full') {
        // Full analysis: ML + rules + fusion
        const payload = buildPayload();
        res = await analyzeFraud(payload, threshold);
        res._mode = 'full';
      } else {
        res = await predictFraud(buildPayload(), threshold);
        res._mode = 'ml_only';
      }
      res.model_used = res.model_used || selectedModel;
      setResult(res);
      setAnalysisForm({ ...form });
    } catch (err) {
      setError(err.message);
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

  /* Determine display risk class */
  const riskLevel = result?.risk_level || '';
  const riskClass = riskLevel === 'HIGH' ? 'danger' : riskLevel === 'MEDIUM' ? 'warning' : 'safe';

  const hourOfDay = Math.floor(form.transaction_time / 3600);

  const analysis = result && analysisForm && result._mode === 'ml_only'
    ? generateAnalysisSummary(analysisForm, result)
    : null;

  /* For full-analysis mode */
  const isFull = result && result._mode === 'full';
  const fraudProb = isFull ? result.final_risk_score : result?.fraud_probability;

  return (
    <div>
      <div className="page-header">
        <h2>Fraud Prediction</h2>
        <p>Analyze a transaction for fraud risk using deep learning + rule engine</p>
      </div>

      {/* Mode Selector */}
      <div className="model-selector" style={{ marginBottom: 'var(--space-sm)' }}>
        <span className="model-selector-label">Analysis Mode</span>
        <div className="model-selector-pills">
          <button
            className={`model-pill ${analysisMode === 'full' ? 'active' : ''}`}
            onClick={() => setAnalysisMode('full')}
            style={analysisMode === 'full' ? { borderColor: '#10b981' } : {}}
          >
            Full Analysis
            <span className="model-pill-badge">ML + Rules</span>
          </button>
          <button
            className={`model-pill ${analysisMode === 'ml_only' ? 'active' : ''}`}
            onClick={() => setAnalysisMode('ml_only')}
          >
            ML Only
            <span className="model-pill-badge">Legacy</span>
          </button>
        </div>
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
                onChange={handleReceiptUpload}
                disabled={extracting}
              />
              {extracting && (
                <div style={{ fontSize: '0.75rem', color: 'var(--accent-blue)', marginTop: '4px' }}>
                  Extracting features from receipt...
                </div>
              )}
            </div>
            <div className="form-group">
              <label className="form-label">Device ID</label>
              <input
                type="text"
                className="form-input"
                value={form.device_id}
                onChange={(e) => handleChange('device_id', e.target.value)}
              />
            </div>
            </div>

          {/* --- IP Metadata Section (collapsible) --- */}
          {analysisMode === 'full' && (
            <div style={{ marginTop: 'var(--space-md)' }}>
              <button
                className="btn btn-outline"
                style={{ width: '100%', fontSize: '0.8rem', padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                onClick={() => setShowIpFields(!showIpFields)}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b' }}></span>
                  IP Metadata (optional)
                </span>
                <span>{showIpFields ? '▲' : '▼'}</span>
              </button>
              {showIpFields && (
                <div className="grid-2" style={{ gap: 'var(--space-md)', marginTop: 'var(--space-md)', padding: 'var(--space-md)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                  <div className="form-group">
                    <label className="form-label">IP Country Code</label>
                    <input
                      type="text"
                      className="form-input"
                      placeholder="e.g. IN, US, GB"
                      value={form.ip_country}
                      onChange={(e) => handleChange('ip_country', e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">IP Geo Distance (km)</label>
                    <input
                      type="number"
                      className="form-input"
                      placeholder="Distance from registered city"
                      value={form.ip_geo_distance_km}
                      onChange={(e) => handleChange('ip_geo_distance_km', e.target.value)}
                      min="0"
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">IP Login Count (24h)</label>
                    <input
                      type="number"
                      className="form-input"
                      placeholder="Accounts on this IP"
                      value={form.ip_login_count}
                      onChange={(e) => handleChange('ip_login_count', e.target.value)}
                      min="0"
                    />
                  </div>
                  <div className="form-group" style={{ display: 'flex', gap: 'var(--space-lg)', alignItems: 'center', marginTop: '12px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem' }}>
                      <input
                        type="checkbox"
                        checked={form.ip_is_vpn}
                        onChange={(e) => handleChange('ip_is_vpn', e.target.checked)}
                      />
                      VPN Detected
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem' }}>
                      <input
                        type="checkbox"
                        checked={form.ip_is_proxy}
                        onChange={(e) => handleChange('ip_is_proxy', e.target.checked)}
                      />
                      Proxy Detected
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* --- Small-Amount Pattern Section (collapsible) --- */}
          {analysisMode === 'full' && (
            <div style={{ marginTop: 'var(--space-md)' }}>
              <button
                className="btn btn-outline"
                style={{ width: '100%', fontSize: '0.8rem', padding: '8px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                onClick={() => setShowPatternFields(!showPatternFields)}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ec4899' }}></span>
                  Pattern Context (optional)
                </span>
                <span>{showPatternFields ? '▲' : '▼'}</span>
              </button>
              {showPatternFields && (
                <div className="grid-2" style={{ gap: 'var(--space-md)', marginTop: 'var(--space-md)', padding: 'var(--space-md)', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)' }}>
                  <div className="form-group">
                    <label className="form-label">Small Txns in 6h ({"<"} ₹200)</label>
                    <input
                      type="number"
                      className="form-input"
                      placeholder="Count of small txns"
                      value={form.small_txn_count_6h}
                      onChange={(e) => handleChange('small_txn_count_6h', e.target.value)}
                      min="0"
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Cumulative Amount 6h (₹)</label>
                    <input
                      type="number"
                      className="form-input"
                      placeholder="Total small amounts"
                      value={form.cumulative_amount_6h}
                      onChange={(e) => handleChange('cumulative_amount_6h', e.target.value)}
                      min="0"
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Same Merchant Hits (1h)</label>
                    <input
                      type="number"
                      className="form-input"
                      placeholder="Repeat merchant count"
                      value={form.same_merchant_count_1h}
                      onChange={(e) => handleChange('same_merchant_count_1h', e.target.value)}
                      min="0"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

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
                {analysisMode === 'full' ? 'Running Full Analysis...' : `Analyzing with ${selectedModel === 'lstm' ? 'LSTM' : 'MLP'}...`}
              </>
            ) : (
              <>
                {analysisMode === 'full'
                  ? 'Run Full Analysis (ML + Rules)'
                  : `Analyze Transaction (${selectedModel === 'lstm' ? 'LSTM' : 'MLP'})`}
              </>
            )}
          </button>
        </div>

        {/* Result */}
        <div className="card animate-in animate-in-delay-1">
          <div className="card-header">
            <span className="card-title">Analysis Result</span>
            {result && (
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                {isFull && (
                  <span style={{
                    padding: '2px 8px', borderRadius: '99px', fontSize: '0.65rem',
                    fontWeight: 600, background: 'rgba(16,185,129,0.12)', color: '#10b981',
                  }}>ML+Rules</span>
                )}
                <span className={`model-tag ${result.model_used}`}>
                  {result.model_used === 'lstm' ? 'LSTM' : 'MLP'}
                </span>
              </div>
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
                {((fraudProb || 0) * 100).toFixed(1)}%
              </div>
              <span className={`risk-badge ${riskLevel.toLowerCase()}`}>
                {riskLevel} RISK
              </span>

              <div className="result-details">
                {isFull ? (
                  <>
                    <div className="result-detail">
                      <div className="result-detail-label">Final Score</div>
                      <div className="result-detail-value">{(result.final_risk_score * 100).toFixed(2)}%</div>
                    </div>
                    <div className="result-detail">
                      <div className="result-detail-label">ML Score</div>
                      <div className="result-detail-value" style={{ color: '#3b82f6' }}>{(result.ml_score * 100).toFixed(2)}%</div>
                    </div>
                    <div className="result-detail">
                      <div className="result-detail-label">Rule Score</div>
                      <div className="result-detail-value" style={{ color: '#f59e0b' }}>{(result.rule_score * 100).toFixed(2)}%</div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="result-detail">
                      <div className="result-detail-label">Probability</div>
                      <div className="result-detail-value">{(result.fraud_probability * 100).toFixed(2)}%</div>
                    </div>
                    <div className="result-detail">
                      <div className="result-detail-label">Risk Level</div>
                      <div className="result-detail-value">{riskLevel}</div>
                    </div>
                    <div className="result-detail">
                      <div className="result-detail-label">Model</div>
                      <div className="result-detail-value">{result.model_used === 'lstm' ? 'LSTM' : 'MLP (DNN)'}</div>
                    </div>
                  </>
                )}
              </div>

              {/* Score Breakdown Bar (full analysis) */}
              {isFull && (
                <div style={{ marginTop: 'var(--space-lg)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Score Composition</div>
                  <div style={{ display: 'flex', borderRadius: '6px', overflow: 'hidden', height: '14px', background: 'var(--bg-tertiary)' }}>
                    <div style={{
                      width: `${Math.min(100, result.ml_score * 100 * 0.6)}%`,
                      background: 'linear-gradient(90deg, #3b82f6, #60a5fa)',
                      transition: 'width 0.8s ease',
                    }}></div>
                    <div style={{
                      width: `${Math.min(100 - result.ml_score * 100 * 0.6, result.rule_score * 100)}%`,
                      background: 'linear-gradient(90deg, #f59e0b, #fbbf24)',
                      transition: 'width 0.8s ease',
                    }}></div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    <span style={{ color: '#3b82f6' }}>● ML ({(result.ml_score * 100).toFixed(1)}%)</span>
                    <span style={{ color: '#f59e0b' }}>● Rules ({(result.rule_score * 100).toFixed(1)}%)</span>
                  </div>
                </div>
              )}

              {/* Probability gauge (ML-only mode) */}
              {!isFull && (
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
              )}

              {/* Fraud type indicators (full mode) */}
              {isFull && result.fraud_type_indicators && result.fraud_type_indicators.length > 0 && (
                <div style={{ marginTop: 'var(--space-lg)', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {result.fraud_type_indicators.map((type) => {
                    const meta = CATEGORY_META[type] || { label: type, color: '#6b7280', bg: 'rgba(107,114,128,0.12)' };
                    return (
                      <span key={type} style={{
                        padding: '3px 10px', borderRadius: '99px', fontSize: '0.7rem',
                        fontWeight: 600, background: meta.bg, color: meta.color,
                      }}>
                        {meta.label}
                      </span>
                    );
                  })}
                </div>
              )}

              <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginTop: 'var(--space-lg)' }}>
                {result.message}
              </p>
            </div>
          )}

          {!result && !error && (
            <div className="empty-state">
              <p>Submit a transaction to see the analysis result</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                {analysisMode === 'full'
                  ? 'Full Analysis combines ML model with IP & pattern rules'
                  : 'Choose MLP or LSTM model above, then analyze the transaction'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* --- Signal Breakdown (Full Analysis Mode) --- */}
      {isFull && (result.signals?.length > 0 || result.positive_signals?.length > 0) && (
        <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
          <div className="card-header">
            <span className="card-title">
              Signal Breakdown — {result.signals?.length || 0} risk signal{result.signals?.length !== 1 ? 's' : ''} detected
            </span>
          </div>

          {/* Triggered risk signals */}
          {result.signals && result.signals.length > 0 && (
            <div className="analysis-section">
              <div className="analysis-section-header risk">
                <span>Risk Signals ({result.signals.length})</span>
              </div>
              <div className="analysis-factors-list">
                {result.signals.map((sig, i) => {
                  const meta = CATEGORY_META[sig.category] || { label: sig.category, color: '#6b7280', bg: 'rgba(107,114,128,0.12)' };
                  const sevClass = sig.severity >= 0.6 ? 'high' : sig.severity >= 0.3 ? 'medium' : 'low';
                  return (
                    <div key={i} className={`analysis-factor-item ${sevClass}`}>
                      <div className="analysis-factor-content">
                        <div className="analysis-factor-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            padding: '1px 6px', borderRadius: '4px', fontSize: '0.6rem',
                            fontWeight: 600, background: meta.bg, color: meta.color,
                          }}>{meta.label}</span>
                          {sig.rule_id.replace(/_/g, ' ')}
                        </div>
                        <div className="analysis-factor-detail">{sig.explanation}</div>
                        <div style={{ display: 'flex', gap: '12px', marginTop: '4px', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          <span>Severity: {(sig.severity * 100).toFixed(0)}%</span>
                          <span>Weight: {(sig.weight * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <span className={`analysis-severity-badge ${sevClass}`}>
                        {sevClass}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Positive signals (non-triggered) */}
          {result.positive_signals && result.positive_signals.length > 0 && (
            <div className="analysis-section">
              <div className="analysis-section-header safe">
                <span>Clear Signals ({result.positive_signals.length})</span>
              </div>
              <div className="analysis-factors-list">
                {result.positive_signals.map((sig, i) => {
                  const meta = CATEGORY_META[sig.category] || { label: sig.category, color: '#6b7280', bg: 'rgba(107,114,128,0.12)' };
                  return (
                    <div key={i} className="analysis-factor-item positive">
                      <div className="analysis-factor-content">
                        <div className="analysis-factor-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            padding: '1px 6px', borderRadius: '4px', fontSize: '0.6rem',
                            fontWeight: 600, background: meta.bg, color: meta.color,
                          }}>{meta.label}</span>
                          {sig.rule_id.replace(/_/g, ' ')}
                        </div>
                        <div className="analysis-factor-detail">{sig.explanation}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Analysis Summary - shows after ML-only prediction */}
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
            <div className="form-group">
              <label className="form-label">Device ID</label>
              <input
                type="text"
                className="form-input"
                value={form.device_id}
                onChange={(e) => handleChange('device_id', e.target.value)}
              />
            </div>
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
                ip_country: 'IN', ip_is_vpn: false, ip_is_proxy: false,
                ip_login_count: '1', ip_geo_distance_km: '10',
                small_txn_count_6h: '0', cumulative_amount_6h: '50', same_merchant_count_1h: '0',
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
                ip_country: 'RU', ip_is_vpn: true, ip_is_proxy: false,
                ip_login_count: '5', ip_geo_distance_km: '4500',
                small_txn_count_6h: '0', cumulative_amount_6h: '', same_merchant_count_1h: '0',
              });
              setResult(null);
              setAnalysisForm(null);
            }}
          >
            🚨 IP Anomaly
          </button>
          <button
            className="btn btn-outline"
            onClick={() => {
              setForm({
                transaction_amount: 35,
                transaction_time: 54000,
                location: 'Bengaluru',
                device_id: 'DEV050',
                merchant_category: 'online',
                account_age_days: 200,
                transaction_count_24h: 18,
                avg_transaction_amount: 150,
                ip_country: 'IN', ip_is_vpn: true, ip_is_proxy: false,
                ip_login_count: '2', ip_geo_distance_km: '100',
                small_txn_count_6h: '12', cumulative_amount_6h: '1800', same_merchant_count_1h: '5',
              });
              setResult(null);
              setAnalysisForm(null);
            }}
          >
            🔢 Card Testing
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
                ip_country: '', ip_is_vpn: false, ip_is_proxy: false,
                ip_login_count: '', ip_geo_distance_km: '',
                small_txn_count_6h: '', cumulative_amount_6h: '', same_merchant_count_1h: '',
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
