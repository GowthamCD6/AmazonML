import React, { useState } from 'react';
import { Search, Scale, Layers, FileText } from 'lucide-react';
import Navbar from './components/Navbar';
import StatsOverview from './components/StatsOverview';
import LiveResolver from './components/LiveResolver';
import PairwiseInspector from './components/PairwiseInspector';
import ExperimentsTable from './components/ExperimentsTable';
import ReportsViewer from './components/ReportsViewer';

export default function App() {
  const [activeTab, setActiveTab] = useState('resolver');

  const navTabs = [
    { key: 'resolver', label: 'Live Entity Matcher', icon: Search },
    { key: 'inspector', label: 'Pairwise Deep Inspector', icon: Scale },
    { key: 'experiments', label: 'Experiments & Leaderboard', icon: Layers },
    { key: 'reports', label: 'Technical Documentation', icon: FileText }
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="app-container" style={{ flex: 1, padding: '28px 24px' }}>
        <StatsOverview />

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '24px' }}>
          {navTabs.map(t => {
            const Icon = t.icon;
            const isActive = activeTab === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key)}
                className={isActive ? 'btn-gradient' : 'btn-ghost'}
                style={{ borderRadius: '10px', fontSize: '0.9rem', padding: '10px 18px' }}
              >
                <Icon size={16} />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        {activeTab === 'resolver' && <LiveResolver />}
        {activeTab === 'inspector' && <PairwiseInspector />}
        {activeTab === 'experiments' && <ExperimentsTable />}
        {activeTab === 'reports' && <ReportsViewer />}
      </main>

      <footer style={{ borderTop: '1px solid var(--border-subtle)', padding: '24px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        <div className="app-container">
          Amazon ML Challenge 2026 — Business Entity Resolution Production Solution • React + Vite Frontend & FastAPI Backend
        </div>
      </footer>
    </div>
  );
}
