import React, { useState } from 'react';
import { Scale, Sparkles, CheckCircle2, XCircle } from 'lucide-react';
import { comparePair } from '../services/api';

export default function PairwiseInspector() {
  const [e1Name, setE1Name] = useState('Clearwater Analytics Pvt. Ltd.');
  const [e1Addr, setE1Addr] = useState('450 Market Street, Suite 200, San Francisco, 94105');
  const [e1Country, setE1Country] = useState('US');

  const [e2Name, setE2Name] = useState('Clearwater Analytics');
  const [e2Addr, setE2Addr] = useState('450 Market St #200, San Francisco, 94105, US');
  const [e2Country, setE2Country] = useState('US');

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleCompare = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await comparePair(
        { business_name: e1Name, business_address: e1Addr, country: e1Country },
        { business_name: e2Name, business_address: e2Addr, country: e2Country },
        0.30
      );
      setResult(data);
    } catch (err) {
      alert("Comparison error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="glass-card">
        <h3 style={{ fontSize: '1.2rem', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Scale size={20} color="var(--accent-cyan)" /> Deep Pairwise Feature & Similarity Inspector
        </h3>

        <form onSubmit={handleCompare}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '20px' }}>
            {/* Record 1 */}
            <div style={{ background: 'rgba(10, 14, 26, 0.5)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '12px' }}>Reference Entity (Source 1)</div>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Business Name</label>
                <input type="text" value={e1Name} onChange={e => setE1Name(e.target.value)} required />
              </div>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Address</label>
                <input type="text" value={e1Addr} onChange={e => setE1Addr(e.target.value)} required />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Country</label>
                <input type="text" value={e1Country} onChange={e => setE1Country(e.target.value)} maxLength={4} />
              </div>
            </div>

            {/* Record 2 */}
            <div style={{ background: 'rgba(10, 14, 26, 0.5)', padding: '16px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: '700', color: 'var(--accent-green)', marginBottom: '12px' }}>Candidate Entity (Source 2/3)</div>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Business Name</label>
                <input type="text" value={e2Name} onChange={e => setE2Name(e.target.value)} required />
              </div>
              <div style={{ marginBottom: '10px' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Address</label>
                <input type="text" value={e2Addr} onChange={e => setE2Addr(e.target.value)} required />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Country</label>
                <input type="text" value={e2Country} onChange={e => setE2Country(e.target.value)} maxLength={4} />
              </div>
            </div>
          </div>

          <button type="submit" className="btn-gradient" disabled={loading}>
            <Sparkles size={18} />
            <span>{loading ? 'Computing Features...' : 'Compare Business Pair'}</span>
          </button>
        </form>
      </div>

      {result && (
        <div className="glass-card">
          <div style={{
            background: result.predicted_same_business ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
            border: `1px solid ${result.predicted_same_business ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            padding: '20px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              {result.predicted_same_business ? <CheckCircle2 size={32} color="var(--accent-green)" /> : <XCircle size={32} color="var(--accent-rose)" />}
              <div>
                <h4 style={{ fontSize: '1.2rem', color: result.predicted_same_business ? 'var(--accent-green)' : 'var(--accent-rose)' }}>
                  {result.predicted_same_business ? 'MATCH CONFIRMED: Same Real-World Business' : 'NON-MATCH: Different Businesses'}
                </h4>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Model Decision Threshold: {result.decision_threshold}
                </div>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Match Probability</div>
              <div style={{ fontSize: '1.8rem', fontWeight: '800', color: result.predicted_same_business ? 'var(--accent-green)' : 'var(--accent-rose)' }}>
                {(result.match_probability * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          <h4 style={{ fontSize: '1.05rem', marginBottom: '14px' }}>Computed Pairwise Feature Matrix (41 Signals)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ background: 'rgba(10, 14, 26, 0.6)', borderBottom: '1px solid var(--border-subtle)' }}>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Feature Signal</th>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Category</th>
                  <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Calculated Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.features).map(([k, v]) => {
                  let cat = "Cross-Field";
                  if (k.startsWith("f_name")) cat = "Name Similarity";
                  else if (k.startsWith("f_addr")) cat = "Address Similarity";
                  else if (k.startsWith("f_country")) cat = "Country";
                  else if (k.startsWith("f_block")) cat = "Blocking Signal";

                  return (
                    <tr key={k} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                      <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{k}</td>
                      <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>{cat}</td>
                      <td style={{ padding: '10px 14px', fontWeight: '700', color: '#fff' }}>{v}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
