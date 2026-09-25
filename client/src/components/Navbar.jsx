import React, { useState, useEffect } from 'react';
import { Sparkles, Database, Layers, CheckCircle } from 'lucide-react';
import { fetchHealth } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [health, setHealth] = useState({ status: 'connecting', count: 0 });

  useEffect(() => {
    fetchHealth()
      .then(d => setHealth({ status: 'connected', count: d.target_records_count }))
      .catch(() => setHealth({ status: 'offline', count: 0 }));
  }, []);

  return (
    <header style={{
      background: 'rgba(10, 14, 26, 0.85)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border-subtle)',
      position: 'sticky',
      top: 0,
      zIndex: 50,
      padding: '16px 0'
    }}>
      <div className="app-container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, var(--primary), var(--accent-cyan))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px var(--primary-glow)',
            color: '#fff',
            fontWeight: '800',
            fontSize: '20px'
          }}>
            ⚡
          </div>
          <div>
            <h1 style={{ fontSize: '1.2rem', fontWeight: '800', color: '#fff' }}>Amazon ML Challenge 2026</h1>
            <div style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)', fontWeight: '600', letterSpacing: '0.06em' }}>
              BUSINESS ENTITY RESOLUTION ENGINE
            </div>
          </div>
        </div>

        {/* Backend Status Badge */}
        <div className={`badge-pill ${health.status === 'connected' ? 'badge-green' : 'badge-rose'}`}>
          {health.status === 'connected' && <div className="pulse-indicator"></div>}
          <span>{health.status === 'connected' ? `Model Online (${health.count.toLocaleString()} Targets)` : 'Backend Offline'}</span>
        </div>
      </div>
    </header>
  );
}
