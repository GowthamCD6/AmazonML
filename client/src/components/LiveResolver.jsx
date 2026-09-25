import React, { useState } from 'react';
import { Search, Sparkles, MapPin, Globe, Check, AlertCircle, RefreshCw } from 'lucide-react';
import { resolveEntity } from '../services/api';

export default function LiveResolver() {
  const [name, setName] = useState('Apex Logistics 128 Pty Ltd');
  const [address, setAddress] = useState('4820 George Street, Sydney, 2000, AU');
  const [country, setCountry] = useState('AU');
  const [threshold, setThreshold] = useState(0.30);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleResolve = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await resolveEntity({
        business_name: name,
        business_address: address,
        country: country
      }, threshold);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (presetName) => {
    if (presetName === 'apex') {
      setName('Apex Logistics 128 Pty Ltd');
      setAddress('4820 George Street, Sydney, 2000, AU');
      setCountry('AU');
    } else if (presetName === 'clearwater') {
      setName('Clearwater Analytics Inc.');
      setAddress('123 Main St, Suite 400, New York, 10001');
      setCountry('US');
    } else if (presetName === 'singleton') {
      setName('Totally Unique NonExistent Business Entity 9999');
      setAddress('9999 Nowhere Land Road, Mars Colony');
      setCountry('US');
    }
  };

  return (
    <div>
      {/* Input Card */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search size={20} color="var(--primary)" /> Real-Time Query Business Entity Matcher
          </h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button type="button" className="btn-ghost" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadPreset('apex')}>Sample AU</button>
            <button type="button" className="btn-ghost" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadPreset('clearwater')}>Sample US</button>
            <button type="button" className="btn-ghost" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => loadPreset('singleton')}>Sample Singleton</button>
          </div>
        </div>

        <form onSubmit={handleResolve}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Business Name *</label>
              <input type="text" value={name} onChange={e => setName(e.target.value)} required />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Business Address *</label>
              <input type="text" value={address} onChange={e => setAddress(e.target.value)} required />
            </div>

            <div style={{ maxWidth: '160px' }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '6px' }}>Country Code</label>
              <input type="text" value={country} onChange={e => setCountry(e.target.value)} maxLength={4} />
            </div>

            <div style={{ maxWidth: '200px' }}>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                F0.5 Threshold: <strong style={{ color: 'var(--accent-cyan)' }}>{threshold.toFixed(2)}</strong>
              </label>
              <input type="range" min="0.10" max="0.95" step="0.05" value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} />
            </div>
          </div>

          <button type="submit" className="btn-gradient" disabled={loading}>
            {loading ? <RefreshCw className="spin" size={18} /> : <Sparkles size={18} />}
            <span>{loading ? 'Resolving Query...' : 'Run Entity Resolution'}</span>
          </button>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="glass-card" style={{ borderColor: 'var(--accent-rose)', color: 'var(--accent-rose)' }}>
          <AlertCircle size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
          <span>{error}</span>
        </div>
      )}

      {/* Results Card */}
      {result && (
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
            <div>
              <h4 style={{ fontSize: '1.1rem' }}>Resolution Results</h4>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Evaluated {result.candidates_count} candidate pairs in <strong style={{ color: 'var(--accent-cyan)' }}>{result.latency_ms}ms</strong>
              </div>
            </div>

            <div className={`badge-pill ${result.is_singleton ? 'badge-rose' : 'badge-green'}`}>
              {result.is_singleton ? 'Singleton (0 Matches)' : `${result.matches_count} Match(es) Found`}
            </div>
          </div>

          {result.matches.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-muted)' }}>
              <p style={{ fontSize: '1.1rem', marginBottom: '4px' }}>No candidate scored above threshold ({result.threshold_applied.toFixed(2)})</p>
              <p style={{ fontSize: '0.85rem' }}>This entity safely defaults to an empty match list <code>[]</code> (full credit on singletons).</p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {result.matches.map((m, idx) => (
                <div key={idx} style={{
                  background: m.is_match ? 'rgba(16, 185, 129, 0.05)' : 'rgba(10, 14, 26, 0.5)',
                  border: `1px solid ${m.is_match ? 'rgba(16, 185, 129, 0.3)' : 'var(--border-subtle)'}`,
                  borderLeft: `4px solid ${m.is_match ? 'var(--accent-green)' : 'var(--border-subtle)'}`,
                  borderRadius: '12px',
                  padding: '16px 20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <strong style={{ fontSize: '1.05rem', color: '#fff' }}>{m.business_name}</strong>
                      <span style={{ fontSize: '0.75rem', background: 'rgba(99, 102, 241, 0.15)', color: 'var(--accent-cyan)', padding: '2px 8px', borderRadius: '4px', fontFamily: 'var(--font-mono)' }}>
                        {m.candidate_entity_id}
                      </span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>• {m.source}</span>
                    </div>

                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <MapPin size={14} /> {m.business_address} {m.country && `• 🌍 ${m.country}`}
                    </div>

                    {/* Blocking tags */}
                    <div style={{ marginTop: '8px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Blocking:</span>
                      {Object.entries(m.blocking_signals).filter(([_, v]) => v === 1).map(([k]) => (
                        <span key={k} style={{ fontSize: '0.7rem', background: 'rgba(255, 255, 255, 0.06)', padding: '1px 6px', borderRadius: '4px', color: 'var(--text-muted)' }}>
                          {k}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right', minWidth: '150px' }}>
                    <div style={{ fontSize: '0.8rem', fontWeight: '700', color: m.is_match ? 'var(--accent-green)' : 'var(--text-muted)', marginBottom: '2px' }}>
                      {m.is_match ? '✅ MATCH' : '❌ REJECTED'}
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '800', color: m.is_match ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                      {(m.match_probability * 100).toFixed(1)}%
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '9999px', overflow: 'hidden', marginTop: '4px' }}>
                      <div style={{
                        width: `${m.match_probability * 100}%`,
                        height: '100%',
                        background: m.is_match ? 'linear-gradient(90deg, var(--accent-cyan), var(--accent-green))' : 'var(--text-dim)'
                      }}></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
