import { useState } from 'react';
import { predictFraud, predictFraudLSTM, predictFromReceipt } from '../api';

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
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = receiptFile
        ? await predictFromReceipt(form, receiptFile, selectedModel)
        : await (selectedModel === 'lstm' ? predictFraudLSTM : predictFraud)(form);
      setResult({ ...res, model_used: selectedModel });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const hourOfDay = Math.floor(form.transaction_time / 3600);

  return (
    <div>
      <h2>Fraud Prediction</h2>
      <div>
        <button onClick={() => setSelectedModel('mlp')}>MLP</button>
        <button onClick={() => setSelectedModel('lstm')}>LSTM</button>
      </div>
      <div>
        <input type="number" value={form.transaction_amount} onChange={e => handleChange('transaction_amount', parseFloat(e.target.value)||0)} placeholder="Amount" />
        <input type="range" min="0" max="86399" value={form.transaction_time} onChange={e => handleChange('transaction_time', parseInt(e.target.value))} />
        <input type="file" onChange={e => setReceiptFile(e.target.files[0])} />
        <button onClick={handleSubmit} disabled={loading}>Analyze</button>
      </div>
      {error && <div>{error}</div>}
      {result && <div>{result.is_fraud ? 'Fraud' : 'Legit'} - {result.fraud_probability}</div>}
    </div>
  );
}
