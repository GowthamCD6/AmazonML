import React, { useEffect, useState } from 'react';
import { FileText } from 'lucide-react';
import { marked } from 'marked';
import { fetchReport } from '../services/api';

export default function ReportsViewer() {
  const [activeReport, setActiveReport] = useState('validation_report');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);

  const reports = [
    { key: 'validation_report', label: '📊 Validation Report' },
    { key: 'data_profile', label: '📈 Dataset Profile' },
    { key: 'error_analysis', label: '⚠️ Error Analysis' },
    { key: 'methodology', label: '📘 Methodology Document' },
    { key: 'readme', label: '📖 README' },
  ];

  useEffect(() => {
    setLoading(true);
    fetchReport(activeReport)
      .then(d => setContent(marked.parse(d.content)))
      .catch(err => setContent(`<p style="color: var(--accent-rose);">Failed to load report: ${err.message}</p>`))
      .finally(() => setLoading(false));
  }, [activeReport]);

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', flexWrap: 'wrap' }}>
        {reports.map(r => (
          <button
            key={r.key}
            className={activeReport === r.key ? 'btn-gradient' : 'btn-ghost'}
            style={{ padding: '8px 16px', fontSize: '0.85rem' }}
            onClick={() => setActiveReport(r.key)}
          >
            {r.label}
          </button>
        ))}
      </div>

      <div style={{
        background: 'rgba(10, 14, 26, 0.6)',
        padding: '24px',
        borderRadius: '12px',
        border: '1px solid var(--border-subtle)',
        minHeight: '400px'
      }}>
        {loading ? (
          <p style={{ color: 'var(--text-muted)' }}>Loading report contents...</p>
        ) : (
          <div className="markdown-content" dangerouslySetInnerHTML={{ __html: content }} />
        )}
      </div>
    </div>
  );
}
