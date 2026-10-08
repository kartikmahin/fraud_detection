import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

export default function Sidebar({ activePage, onNavigate, isTraining, theme, onToggleTheme }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    
    const interval = setInterval(() => {
      checkHealth()
        .then(setHealth)
        .catch(() => setHealth(null));
    }, 10000);
    
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'predict', label: 'Predict' },
    { id: 'training', label: 'Training' },
    { id: 'metrics', label: 'Model Metrics' },
    { id: 'about', label: 'About' },
  ];

  const isOnline = health !== null;
  const mlpLoaded = health?.model_loaded;
  const lstmLoaded = health?.lstm_model_loaded;
  const isDark = theme === 'dark';

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <h1>FraudShield AI</h1>
        <div className="subtitle">PyTorch Deep Learning</div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''} ${item.id === 'training' && isTraining ? 'training-active' : ''}`}
            onClick={() => onNavigate(item.id)}
          >
            <span>{item.label}</span>
            {item.id === 'training' && isTraining && (
              <span className="nav-training-badge">
                <span className="nav-training-dot"></span>
                Running
              </span>
            )}
          </div>
        ))}
      </nav>

      {/* Theme Toggle */}
      <div className="theme-toggle-container">
        <div className="theme-toggle" onClick={onToggleTheme}>
          <div className="theme-toggle-track">
            <div className="theme-toggle-thumb"></div>
          </div>
          <span className="theme-toggle-label">{isDark ? 'Night Mode' : 'Day Mode'}</span>
        </div>
      </div>

      <div className="sidebar-footer">
        {/* API status badge removed */}
        {isOnline && (
          <>
            <div className="status-badge" style={{ marginTop: '6px' }}>
              <span className={`status-dot ${mlpLoaded ? '' : 'offline'}`}></span>
              <span>{mlpLoaded ? 'MLP Model ✓' : 'MLP Not Loaded'}</span>
            </div>
            <div className="status-badge" style={{ marginTop: '4px' }}>
              <span className={`status-dot ${lstmLoaded ? '' : 'offline'}`}></span>
              <span>{lstmLoaded ? 'LSTM Model ✓' : 'LSTM Not Loaded'}</span>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

