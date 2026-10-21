import { useState, useRef } from 'react';
import { fileComplaint } from '../api';

export default function Complaint() {
  const [description, setDescription] = useState('');
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [fraudAmount, setFraudAmount] = useState('');
  const [transactionDate, setTransactionDate] = useState('');
  const [bankName, setBankName] = useState('');
  const [upiId, setUpiId] = useState('');
  const [merchantName, setMerchantName] = useState('');
  const [transactionId, setTransactionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);
  const recognitionRef = useRef(null);

  // Check if speech recognition is supported
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  const startListening = () => {
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      setError('Speech recognition is not supported in this browser. Please use Chrome or Edge, or type your complaint manually.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    let finalTranscript = description;

    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }
      setDescription(finalTranscript + interimTranscript);
    };

    recognition.onerror = (event) => {
      if (event.error === 'not-allowed') {
        setError('Microphone access denied. Click the lock icon in your browser address bar, allow microphone access, and try again.');
      } else if (event.error === 'no-speech') {
        setError('No speech detected. Please speak clearly and try again, or type your complaint manually.');
      } else if (event.error === 'network') {
        setError('Speech recognition requires internet access to Google servers. If you are on a VPN, corporate network, or firewall, the service may be blocked. Try: 1) Disable VPN  2) Use a different network  3) Type your complaint instead.');
      } else if (event.error === 'aborted') {
        setError('Speech recognition was aborted. Please try again.');
      } else if (event.error === 'audio-capture') {
        setError('No microphone found. Please connect a microphone and try again.');
      } else if (event.error === 'service-not-allowed') {
        setError('Speech service not allowed. Please check browser permissions and try again.');
      } else {
        setError(`Speech recognition error: ${event.error}. Please try typing your complaint instead.`);
      }
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      // Auto-restart for continuous-like behavior
      if (isListening && recognitionRef.current === recognition) {
        try {
          recognition.start();
        } catch (e) {
          // Ignore restart errors
        }
      }
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch (err) {
      setError('Failed to start speech recognition. Please try again.');
      setIsListening(false);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!description.trim()) {
      setError('Please provide a description of the fraud.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await fileComplaint({
        description: description.trim(),
        name: name || null,
        phone: phone || null,
        email: email || null,
        fraud_amount: fraudAmount ? parseFloat(fraudAmount) : null,
        transaction_date: transactionDate || null,
        bank_name: bankName || null,
        upi_id: upiId || null,
        merchant_name: merchantName || null,
        transaction_id: transactionId || null,
      });
      setResult(res);
    } catch (err) {
      const msg = err?.message || err?.toString() || 'Unknown error occurred';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setDescription('');
    setName('');
    setPhone('');
    setEmail('');
    setFraudAmount('');
    setTransactionDate('');
    setBankName('');
    setUpiId('');
    setMerchantName('');
    setTransactionId('');
    setResult(null);
    setError(null);
    setReceiptFile(null);
  };

  if (result) {
    return (
      <div>
        <div className="page-header">
          <h2>Complaint Filed Successfully</h2>
          <p>Your complaint has been recorded. Complete the official filing on the National Cyber Crime Portal.</p>
        </div>

        <div className="card animate-in" style={{ maxWidth: '700px', margin: '0 auto' }}>
          <div className="card-header">
            <span className="card-title">Complaint Reference</span>
          </div>

          <div style={{ textAlign: 'center', padding: 'var(--space-lg) 0' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Complaint ID</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent-blue)' }}>
              {result.complaint_id}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '12px' }}>Reference Number</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-secondary)' }}>
              {result.reference_number}
            </div>
          </div>

          <div className="form-hint" style={{ textAlign: 'center', marginBottom: 'var(--space-lg)' }}>
            {result.message}
          </div>

          <div style={{ marginTop: 'var(--space-lg)' }}>
            <div className="card-title" style={{ marginBottom: 'var(--space-md)' }}>Next Steps</div>
            <ol style={{ paddingLeft: '20px', color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.8' }}>
              {result.next_steps.map((step, i) => (
                <li key={i}>{step}</li>
              ))}
            </ol>
          </div>

          <div style={{ marginTop: 'var(--space-xl)', display: 'flex', gap: 'var(--space-md)', justifyContent: 'center' }}>
            <a
              href={result.portal_url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary btn-lg"
            >
              Go to Cyber Crime Portal
            </a>
            <button className="btn btn-outline" onClick={handleReset}>
              File Another Complaint
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <h2>File Fraud Complaint</h2>
        <p>Report cyber fraud to the National Cyber Crime Portal. Ensure you are certain the fraud has occurred before filing.</p>
      </div>

      {/* Warning Banner */}
      <div style={{
        background: 'rgba(245, 158, 11, 0.1)',
        border: '1px solid rgba(245, 158, 11, 0.3)',
        borderRadius: 'var(--radius-md)',
        padding: 'var(--space-md) var(--space-lg)',
        marginBottom: 'var(--space-xl)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 'var(--space-md)',
      }}>
        <div style={{ fontSize: '1.2rem', flexShrink: 0, marginTop: '2px' }}>&#9888;</div>
        <div>
          <div style={{ fontWeight: 600, color: 'var(--warning)', marginBottom: '4px' }}>Important Warning</div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
            Only file a complaint if you are <strong>completely certain</strong> that fraud has occurred. Filing false complaints is a punishable offense.
            Before submitting, ensure you have evidence such as transaction screenshots, bank statements, and communication records.
          </div>
        </div>
      </div>

      <div className="card animate-in">
        <div className="card-header">
          <span className="card-title">Complaint Details</span>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Description with Voice Input */}
          <div className="form-group">
            <label className="form-label">
              Describe the Fraud
              <span className="form-label-optional">Required</span>
            </label>
            <div style={{ position: 'relative' }}>
              <textarea
                className="form-input"
                value={description}
                onChange={e => setDescription(e.target.value)}
                placeholder="Describe what happened: What type of fraud? When did it occur? How much money was involved? What was the modus operandi? Any suspicious links, calls, or messages?"
                rows={6}
                style={{ resize: 'vertical', minHeight: '120px' }}
                required
              />
              <button
                type="button"
                onClick={toggleListening}
                style={{
                  position: 'absolute',
                  bottom: '10px',
                  right: '10px',
                  width: '36px',
                  height: '36px',
                  borderRadius: '50%',
                  border: 'none',
                  background: isListening ? 'var(--danger)' : 'var(--accent-blue)',
                  color: 'white',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.3s ease',
                  boxShadow: isListening ? '0 0 12px rgba(239, 68, 68, 0.4)' : '0 2px 8px rgba(59, 130, 246, 0.3)',
                  animation: isListening ? 'pulse-dot 1s infinite' : 'none',
                }}
                title={isListening ? 'Stop recording' : 'Speak your complaint'}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  {isListening ? (
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  ) : (
                    <>
                      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                      <line x1="12" y1="19" x2="12" y2="23" />
                      <line x1="8" y1="23" x2="16" y2="23" />
                    </>
                  )}
                </svg>
              </button>
            </div>
            {isListening && (
              <div style={{ fontSize: '0.75rem', color: 'var(--danger)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--danger)', animation: 'pulse-dot 1s infinite', display: 'inline-block' }}></span>
                Listening... Speak your complaint clearly
              </div>
            )}
            <div className="form-hint">
              {description.length} characters | Tip: Include details like date, amount, bank name, and how the fraud occurred
            </div>
            {!speechSupported && (
              <div className="form-hint" style={{ color: 'var(--warning)', marginTop: '4px' }}>
                Speech recognition is not available in this browser. Please use Chrome or Edge, or type your complaint manually.
              </div>
            )}
            {speechSupported && (
              <div className="form-hint" style={{ color: 'var(--text-muted)', marginTop: '4px', fontSize: '0.7rem' }}>
                Note: Voice input uses Google speech service and requires internet access. If blocked by firewall/VPN, please type your complaint.
              </div>
            )}
          </div>

          {/* Optional Details */}
          <div style={{ marginTop: 'var(--space-lg)', marginBottom: 'var(--space-md)' }}>
            <div className="card-title" style={{ marginBottom: 'var(--space-md)' }}>Additional Details (Optional)</div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Your Name</label>
              <input
                type="text"
                className="form-input"
                value={name}
                onChange={e => setName(e.target.value)}
                placeholder="Full name"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Phone Number</label>
              <input
                type="tel"
                className="form-input"
                value={phone}
                onChange={e => setPhone(e.target.value)}
                placeholder="+91 XXXXX XXXXX"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <input
                type="email"
                className="form-input"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Fraud Amount (INR)</label>
              <input
                type="number"
                className="form-input"
                value={fraudAmount}
                onChange={e => setFraudAmount(e.target.value)}
                placeholder="Amount lost"
                min="0"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Transaction Date</label>
              <input
                type="date"
                className="form-input"
                value={transactionDate}
                onChange={e => setTransactionDate(e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Bank Name</label>
              <input
                type="text"
                className="form-input"
                value={bankName}
                onChange={e => setBankName(e.target.value)}
                placeholder="e.g. SBI, HDFC"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">UPI ID / VPA</label>
              <input
                type="text"
                className="form-input"
                value={upiId}
                onChange={e => setUpiId(e.target.value)}
                placeholder="e.g. fraud@upi"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Transaction ID</label>
              <input
                type="text"
                className="form-input"
                value={transactionId}
                onChange={e => setTransactionId(e.target.value)}
                placeholder="UTR / Transaction ref"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Merchant / Recipient Name</label>
            <input
              type="text"
              className="form-input"
              value={merchantName}
              onChange={e => setMerchantName(e.target.value)}
              placeholder="Name of the person or entity that received the funds"
            />
          </div>

          {/* Error */}
          {error && (
            <div className="prediction-result danger" style={{ margin: 'var(--space-lg) 0' }}>
              <div className="result-label">Error</div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '8px' }}>{error}</p>
            </div>
          )}

          {/* Submit */}
          <div style={{ display: 'flex', gap: 'var(--space-md)', marginTop: 'var(--space-lg)' }}>
            <button
              type="submit"
              className="btn btn-primary btn-lg"
              style={{ flex: 1 }}
              disabled={loading || !description.trim()}
            >
              {loading ? (
                <>
                  <div className="spinner" style={{ width: '18px', height: '18px', borderWidth: '2px' }}></div>
                  Filing Complaint...
                </>
              ) : (
                'File Complaint'
              )}
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={handleReset}
              disabled={loading}
            >
              Clear
            </button>
          </div>
        </form>
      </div>

      {/* Info Section */}
      <div className="card animate-in" style={{ marginTop: 'var(--space-xl)' }}>
        <div className="card-header">
          <span className="card-title">About Cyber Crime Complaints</span>
        </div>
        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.7' }}>
          <p style={{ marginBottom: 'var(--space-md)' }}>
            The National Cyber Crime Reporting Portal (cybercrime.gov.in) is an initiative of the Government of India
            to facilitate reporting of cyber crimes. You can report various types of cyber fraud including:
          </p>
          <ul style={{ paddingLeft: '20px', marginBottom: 'var(--space-md)' }}>
            <li>Financial fraud (unauthorized transactions, UPI fraud, credit/debit card fraud)</li>
            <li>Online shopping fraud</li>
            <li>Phishing and vishing (voice phishing) attacks</li>
            <li>Identity theft and impersonation</li>
            <li>Investment fraud and Ponzi schemes</li>
            <li>Social media fraud and cyber bullying</li>
          </ul>
          <p>
            <strong>Important:</strong> Keep all evidence (screenshots, transaction IDs, bank statements, call records)
            ready before filing on the official portal. You may also contact your bank's customer care immediately
            to report the fraud and request a chargeback.
          </p>
        </div>
      </div>
    </div>
  );
}