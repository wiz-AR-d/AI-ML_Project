import { useState } from 'react';
import './App.css';
import InputPanel from './components/InputPanel';
import OutputPanel from './components/OutputPanel';
import AgentInputPanel from './components/AgentInputPanel';
import ReportPanel from './components/ReportPanel';

// Vite proxy forwards /api → http://localhost:8000 in dev
// Set VITE_API_URL in .env for production (e.g. https://your-app.onrender.com)
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, "");

export default function App() {
  // ── Mode ──────────────────────────────────────────
  const [mode, setMode] = useState('nlp'); // 'nlp' | 'agent'

  // ── Milestone 1 state ─────────────────────────────
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [step, setStep] = useState(0);

  // ── Milestone 2 state ─────────────────────────────
  const [agentReport, setAgentReport] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentError, setAgentError] = useState(null);
  const [agentStep, setAgentStep] = useState(0);
  const [agentQuery, setAgentQuery] = useState('');

  /* ── M1: NLP Analysis ── */
  const advanceStep = () => setStep(s => s + 1);

  const handleAnalyze = async ({ keywords, files, numTopics, summaryLen, useBoW }) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setStep(0);

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

  /* ── M2: Agent Research ── */
  const handleAgentResearch = async (query) => {
    setAgentLoading(true);
    setAgentError(null);
    setAgentReport(null);
    setAgentQuery(query);
    setAgentStep(0);

    // Step 0 → Search, Step 1 → Analyze, Step 2 → Report
    const stepTimings = [0, 5000, 12000];
    const timers = stepTimings.map((delay, i) =>
      setTimeout(() => setAgentStep(i), delay)
    );

    try {
      const form = new FormData();
      form.append('query', query);

      const res = await fetch(`${API_BASE}/api/v1/ai/agent/research`, {
        method: 'POST',
        body: form,
      });

      timers.forEach(t => clearTimeout(t));

      if (!res.ok) {
        const msg = await res.text().catch(() => 'Server error');
        throw new Error(msg || `HTTP ${res.status}`);
      }

      const data = await res.json();

      if (data.status === 'error') {
        throw new Error(data.message || 'Agent returned an error.');
      }

      setAgentReport(data.report);
    } catch (err) {
      timers.forEach(t => clearTimeout(t));
      setAgentError(err.message || 'Agent failed. Check the backend server.');
    } finally {
      setAgentLoading(false);
      setAgentStep(0);
    }
  };

  /* ── Mode switch resets state ── */
  const switchMode = (m) => {
    setMode(m);
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">🔬</div>
          <span className="header-title">Pluto</span>
        </div>

        {/* Mode Switcher */}
        <div className="mode-switcher" role="tablist" aria-label="Application mode">
          <button
            id="mode-nlp-btn"
            role="tab"
            aria-selected={mode === 'nlp'}
            className={`mode-btn ${mode === 'nlp' ? 'active' : ''}`}
            onClick={() => switchMode('nlp')}
          >
            <span>🧬</span> NLP Analysis
            <span className="mode-badge">M1</span>
          </button>
          <button
            id="mode-agent-btn"
            role="tab"
            aria-selected={mode === 'agent'}
            className={`mode-btn ${mode === 'agent' ? 'active agent-active' : ''}`}
            onClick={() => switchMode('agent')}
          >
            <span>🤖</span> Agentic Research
            <span className="mode-badge mode-badge-agent">M2</span>
          </button>
        </div>
      </header>

      {/* ── Main Grid ── */}
      <main className="main">
        {mode === 'nlp' ? (
          <>
            <InputPanel onAnalyze={handleAnalyze} loading={loading} />
            <OutputPanel result={result} loading={loading} error={error} step={step} />
          </>
        ) : (
          <>
            <AgentInputPanel
              onResearch={handleAgentResearch}
              loading={agentLoading}
              agentStep={agentStep}
            />
            <ReportPanel
              report={agentReport}
              loading={agentLoading}
              error={agentError}
              query={agentQuery}
              agentStep={agentStep}
            />
          </>
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <span>Pluto — AI Research System · M1: NLP Pipeline + M2: Agentic AI</span>
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
