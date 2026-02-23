import { useState, useRef, useCallback } from 'react';

const ACCEPTED = '.txt,.pdf,.docx,.md,.csv';

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function InputPanel({ onAnalyze, loading }) {
    const [keywords, setKeywords] = useState([]);
    const [kInput, setKInput] = useState('');
    const [files, setFiles] = useState([]);
    const [dragging, setDragging] = useState(false);
    const [numTopics, setNumTopics] = useState('5');
    const [summaryLen, setSummaryLen] = useState('3');
    const [useBoW, setUseBoW] = useState(false);
    const fileRef = useRef();

    /* ── keyword helpers ─── */
    const addKeyword = () => {
        const v = kInput.trim();
        if (v && !keywords.includes(v)) setKeywords(k => [...k, v]);
        setKInput('');
    };
    const removeKw = kw => setKeywords(k => k.filter(x => x !== kw));
    const onKwKey = e => {
        if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); addKeyword(); }
    };

    /* ── file helpers ─── */
    const addFiles = raw => {
        const next = Array.from(raw).filter(f =>
            !files.some(ex => ex.name === f.name && ex.size === f.size)
        );
        setFiles(f => [...f, ...next]);
    };
    const onDrop = useCallback(e => {
        e.preventDefault(); setDragging(false);
        addFiles(e.dataTransfer.files);
    }, [files]);
    const onDragOver = e => { e.preventDefault(); setDragging(true); };
    const onDragLeave = () => setDragging(false);
    const removeFile = name => setFiles(f => f.filter(x => x.name !== name));

    /* ── submit ─── */
    const canSubmit = keywords.length > 0 && !loading;
    const handleSubmit = () => {
        if (!canSubmit) return;
        onAnalyze({ keywords, files, numTopics: +numTopics, summaryLen: +summaryLen, useBoW });
    };

    return (
        <aside className="panel" style={{ position: 'sticky', top: '80px' }}>
            <div className="panel-header">
                <div className="panel-header-icon" style={{ background: 'var(--accent-soft)' }}>🔍</div>
                <span className="panel-title">Analysis Inputs</span>
            </div>

            <div className="panel-body">

                {/* ── Topic Keywords ── */}
                <div>
                    <div className="field-label"><span>🏷️</span> Research Topic Keywords</div>
                    <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0 0 0.5rem 0' }}>
                        Enter keywords to search 5,000 arXiv research papers
                    </p>
                    <div className="tag-input-row">
                        <input
                            id="kw-input"
                            className="tag-text-input"
                            value={kInput}
                            onChange={e => setKInput(e.target.value)}
                            onKeyDown={onKwKey}
                            placeholder="e.g. machine learning, NLP, neural network…"
                            disabled={loading}
                        />
                        <button className="btn-add-tag" onClick={addKeyword} disabled={!kInput.trim() || loading}>
                            + Add
                        </button>
                    </div>
                    {keywords.length > 0 && (
                        <div className="tag-row" style={{ marginTop: '0.6rem' }}>
                            {keywords.map(kw => (
                                <span key={kw} className="tag">
                                    {kw}
                                    <button className="tag-remove" onClick={() => removeKw(kw)} title="Remove">✕</button>
                                </span>
                            ))}
                        </div>
                    )}
                </div>

                {/* ── Optional File Upload ── */}
                <div>
                    <div className="field-label"><span>📄</span> Upload Documents
                        <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>
                            optional · txt · pdf · docx
                        </span>
                    </div>
                    <label
                        className={`dropzone${dragging ? ' active' : ''}`}
                        onDrop={onDrop}
                        onDragOver={onDragOver}
                        onDragLeave={onDragLeave}
                        htmlFor="file-input"
                    >
                        <div className="dropzone-icon">📂</div>
                        <div className="dropzone-text">
                            <strong>Click to browse</strong> or drag & drop files here
                        </div>
                        <input
                            id="file-input"
                            type="file"
                            multiple
                            accept={ACCEPTED}
                            ref={fileRef}
                            onChange={e => addFiles(e.target.files)}
                            disabled={loading}
                        />
                    </label>
                    {files.length > 0 && (
                        <div className="file-list">
                            {files.map(f => (
                                <div key={f.name} className="file-item">
                                    <span className="file-icon">📝</span>
                                    <span className="file-name" title={f.name}>{f.name}</span>
                                    <span className="file-size">{formatBytes(f.size)}</span>
                                    <button className="file-remove" onClick={() => removeFile(f.name)} title="Remove">✕</button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* ── NLP Settings ── */}
                <div>
                    <div className="field-label"><span>⚙️</span> NLP Settings</div>
                    <div className="settings-row">
                        <div className="setting-item">
                            <span className="setting-label">📊 Feature Extraction</span>
                            <select
                                id="feature-select"
                                className="select-sm"
                                value={useBoW ? 'bow' : 'tfidf'}
                                onChange={e => setUseBoW(e.target.value === 'bow')}
                                disabled={loading}
                            >
                                <option value="tfidf">TF-IDF</option>
                                <option value="bow">Bag-of-Words</option>
                            </select>
                        </div>
                        <div className="setting-item">
                            <span className="setting-label">🗂️ Topic Clusters</span>
                            <select
                                id="topics-select"
                                className="select-sm"
                                value={numTopics}
                                onChange={e => setNumTopics(e.target.value)}
                                disabled={loading}
                            >
                                {[3, 4, 5, 6, 7, 8, 10].map(n => (
                                    <option key={n} value={n}>{n} topics</option>
                                ))}
                            </select>
                        </div>
                        <div className="setting-item">
                            <span className="setting-label">📝 Summary Sentences</span>
                            <select
                                id="summary-select"
                                className="select-sm"
                                value={summaryLen}
                                onChange={e => setSummaryLen(e.target.value)}
                                disabled={loading}
                            >
                                {[2, 3, 4, 5, 6, 8].map(n => (
                                    <option key={n} value={n}>{n} sentences</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>

                {/* ── Submit ── */}
                <button
                    id="analyze-btn"
                    className="btn-analyze"
                    onClick={handleSubmit}
                    disabled={!canSubmit}
                >
                    {loading
                        ? <><div className="spinner" /> Analyzing…</>
                        : <><span>🚀</span> Run NLP Analysis</>}
                </button>

                {keywords.length === 0 && (
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '-0.5rem' }}>
                        Add at least one keyword to search the arXiv research corpus
                    </p>
                )}
            </div>
        </aside>
    );
}
