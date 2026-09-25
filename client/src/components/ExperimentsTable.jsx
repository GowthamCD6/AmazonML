import React, { useEffect, useState } from 'react';
import { Layers, Trophy } from 'lucide-react';
import { fetchExperiments, fetchLeaderboard } from '../services/api';

export default function ExperimentsTable() {
  const [experiments, setExperiments] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);

  useEffect(() => {
    fetchExperiments().then(setExperiments).catch(() => {});
    fetchLeaderboard().then(setLeaderboard).catch(() => {});
  }, []);

  return (
    <div>
      {/* Experiments Log */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.2rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Layers size={20} color="var(--primary)" /> Experiment Run History (`experiments.csv`)
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
          Strict tracking of each version iteration from V1 baseline to V5 final production model.
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ background: 'rgba(10, 14, 26, 0.6)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Version</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Day</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Change Description</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Model</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Threshold</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Cand Recall</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Precision</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Recall</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Val F0.5</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Decision</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '10px 14px', fontWeight: '800', color: 'var(--accent-cyan)' }}>{exp.version}</td>
                  <td style={{ padding: '10px 14px' }}>Day {exp.day}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-main)' }}>{exp.change}</td>
                  <td style={{ padding: '10px 14px', fontFamily: 'var(--font-mono)' }}>{exp.model}</td>
                  <td style={{ padding: '10px 14px' }}>{exp.threshold}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--accent-green)' }}>{(parseFloat(exp.candidate_recall) * 100).toFixed(1)}%</td>
                  <td style={{ padding: '10px 14px' }}>{exp.validation_precision}</td>
                  <td style={{ padding: '10px 14px' }}>{exp.validation_recall}</td>
                  <td style={{ padding: '10px 14px', fontWeight: '800', color: 'var(--accent-amber)' }}>{exp.validation_f05}</td>
                  <td style={{ padding: '10px 14px' }}>
                    <span className="badge-pill badge-green">{exp.decision}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Leaderboard Log */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.2rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Trophy size={20} color="var(--accent-amber)" /> Submissions & Leaderboard Log (`leaderboard.csv`)
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ background: 'rgba(10, 14, 26, 0.6)', borderBottom: '1px solid var(--border-subtle)' }}>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Version</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Day</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Submission Time</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Val F0.5</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Public Score</th>
                <th style={{ padding: '10px 14px', textAlign: 'left', color: 'var(--text-muted)' }}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((lb, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '10px 14px', fontWeight: '800', color: 'var(--accent-cyan)' }}>{lb.submission_version}</td>
                  <td style={{ padding: '10px 14px' }}>Day {lb.day}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>{lb.submission_time}</td>
                  <td style={{ padding: '10px 14px', fontWeight: '800', color: 'var(--accent-amber)' }}>{lb.validation_f05}</td>
                  <td style={{ padding: '10px 14px' }}>{lb.public_score || 'Pending'}</td>
                  <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>{lb.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
