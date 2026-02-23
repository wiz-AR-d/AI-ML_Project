import { useState } from 'react';
import './App.css';
import InputPanel from './components/InputPanel';
import OutputPanel from './components/OutputPanel';

// Vite proxy forwards /api → http://localhost:8000 in dev
// Set VITE_API_URL in .env for production (e.g. https://your-app.onrender.com)
const API_BASE = import.meta.env.VITE_API_URL || '';

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(0);

  const advanceStep = () => {
    setStep(s => s + 1);
  };

  const handleAnalyze = async ({ keywords, files, numTopics, summaryLen, useBoW }) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setStep(0);

    /* Simulate pipeline progress */
    const interval = setInterval(advanceStep, 700);

    try {
      const form = new FormData();
      keywords.forEach(k => form.append('keywords', k));
      form.append('num_topics', numTopics);
      form.append('summary_sentences', summaryLen);
      form.append('use_bow', useBoW ? 'true' : 'false');
      files.forEach(f => form.append('files', f));

      const res = await fetch(`${API_BASE}/api/v1/ml/analyze`, {
        method: 'POST',
        body: form,
      });

      clearInterval(interval);

      if (!res.ok) {
        const msg = await res.text().catch(() => 'Server error');
        throw new Error(msg || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      clearInterval(interval);
      setError(err.message || 'Unknown error. Check the backend server.');
    } finally {
      setLoading(false);
      setStep(0);
    }
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🔬</div>
          <span className="header-title">Pluto</span>
        </div>
        <span className="header-badge">Milestone 1 · NLP Pipeline</span>
      </header>

      {/* ── Main Grid ── */}
      <main className="main">
        <InputPanel onAnalyze={handleAnalyze} loading={loading} />
        <OutputPanel result={result} loading={loading} error={error} step={step} />
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <span>Pluto — Traditional NLP Analysis System · Milestone 1</span>
        <div className="footer-links">
          <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv" target="_blank" rel="noreferrer">
            arXiv Dataset
          </a>
          <a href="https://github.com" target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </footer>
    </div>
  );
}
