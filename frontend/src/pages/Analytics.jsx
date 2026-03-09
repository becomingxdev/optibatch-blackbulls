import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import { BarChart2, TrendingUp, Target, Zap, Leaf, ArrowLeft, Cpu } from 'lucide-react';
import ParticlesBackground from '../components/ParticlesBackground';
import Chatbot from '../components/Chatbot';

/* ── Data ───────────────────────────────────────────────────────────── */

// Chart 1 — Strategy Comparison (Remains constant to show different fixed strategies)
const strategyData = [
  { strategy: '30%', energy: 40.7, co2: 7.1, savings: 109.5 },
  { strategy: '50%', energy: 37.7, co2: 6.3, savings: 122.5 },
  { strategy: '70%', energy: 34.7, co2: 5.5, savings: 135.5 },
  { strategy: '90%', energy: 31.7, co2: 4.7, savings: 148.5 },
];

/* ── Shared tooltip ─────────────────────────────────────────────────── */
const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900/90 border border-white/10 rounded-xl px-4 py-3 shadow-xl text-xs">
      <p className="font-black text-slate-300 mb-2 uppercase tracking-widest">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }} className="font-bold">
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  );
};

/* ── Chart card wrapper ─────────────────────────────────────────────── */
const ChartCard = ({ title, icon: Icon, children }) => (
  <div className="glass-panel p-6 md:p-8 rounded-3xl relative overflow-hidden">
    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent" />
    <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3 mb-8">
      <Icon size={17} className="text-cyan-400" /> {title}
    </h3>
    {children}
  </div>
);

/* ── Summary KPI strip ──────────────────────────────────────────────── */
const KPIStrip = ({ strategy }) => {
  // Calculate dynamic KPIs based on selected strategy
  const multiplier = strategy / 70; // normalize around default 70
  
  return (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
    {[
      { icon: Zap,       color: 'amber',   label: 'Peak Energy Saved',   value: `${(13.5 * multiplier).toFixed(1)} kWh / batch`,  border: 'border-amber-400/20',   bg: 'bg-amber-400/10',   text: 'text-amber-400' },
      { icon: Leaf,      color: 'emerald', label: 'CO₂ Reduction',       value: `↓ ${(33.8 * multiplier).toFixed(1)} %`,           border: 'border-emerald-400/20', bg: 'bg-emerald-400/10', text: 'text-emerald-400' },
      { icon: TrendingUp, color: 'cyan',   label: 'Projected 30-Day ROI', value: `$${(4095 * multiplier).toLocaleString('en-US', {maximumFractionDigits:0})}`,             border: 'border-cyan-400/20',    bg: 'bg-cyan-400/10',    text: 'text-cyan-400' },
    ].map(({ icon: Icon, label, value, border, bg, text }) => (
      <div key={label} className={`glass-panel p-5 rounded-2xl flex items-center gap-5 border ${border}`}>
        <div className={`p-3 rounded-xl ${bg} border ${border}`}>
          <Icon size={22} className={text} />
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest font-black mb-0.5">{label}</p>
          <p className="text-2xl font-black text-white tracking-tighter">{value}</p>
        </div>
      </div>
    ))}
  </div>
  );
};

/* ── Page ───────────────────────────────────────────────────────────── */
const Analytics = () => {
  const navigate = useNavigate();
  const [strategy, setStrategy] = useState(() => Number(localStorage.getItem('optibatch_strategy')) || 70);

  // Monitor updates across tabs/navigation
  useEffect(() => {
    const handleStorage = () => {
      setStrategy(Number(localStorage.getItem('optibatch_strategy')) || 70);
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const radarData = useMemo(() => {
    const s = strategy / 100;
    return [
      { kpi: 'Energy',  current: 72, target: 70 + (25 * (1-s)) },
      { kpi: 'Quality', current: 85, target: 80 + (18 * s) },
      { kpi: 'Speed',   current: 60, target: 60 + (35 * s) },
      { kpi: 'Cost',    current: 78, target: 70 + (20 * (1-s)) },
      { kpi: 'CO₂',     current: 65, target: 65 + (25 * (1-s)) },
      { kpi: 'Uptime',  current: 92, target: 90 + (8 * s) },
    ];
  }, [strategy]);

  const projectionData = useMemo(() => {
    const s = strategy / 100;
    return Array.from({ length: 30 }, (_, i) => ({
      day: `D${i + 1}`,
      baseline:  90 + i * 0.5,
      optimized: 90 + i * 0.5 + (i * 1.8 * (0.5 + s)) + Math.sin(i * 0.4) * 2,
    }));
  }, [strategy]);

  return (
    <div className="bg-[#050505] min-h-screen text-slate-200 font-sans selection:bg-cyan-500/30 relative">
      <ParticlesBackground />

      <div className="max-w-[1600px] mx-auto p-4 md:p-8 pt-8 relative z-10">

        {/* Header */}
        <header className="flex justify-between items-center mb-8 pb-6 border-b border-white/10">
          <div className="flex items-center gap-5">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 text-slate-400 hover:text-cyan-400 transition-colors text-sm font-bold"
            >
              <ArrowLeft size={18} /> Dashboard
            </button>
            <div className="w-px h-6 bg-white/10" />
            <div className="relative">
              <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-40" />
              <div className="relative bg-gradient-to-br from-gray-900 to-black border border-white/10 p-3 rounded-2xl">
                <BarChart2 className="text-cyan-400" size={28} />
              </div>
            </div>
            <div>
              <h1 className="text-3xl font-black tracking-tighter text-white">
                OPTI<span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">ANALYTICS</span>
              </h1>
              <p className="text-[11px] text-slate-500 uppercase tracking-[0.3em] font-bold mt-1">Batch Intelligence Center</p>
            </div>
          </div>
          <div className="hidden md:flex px-5 py-2.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 items-center gap-3">
            <Cpu size={14} className="text-cyan-400" />
            <span className="text-xs font-black text-cyan-400 tracking-widest uppercase">Live Analysis</span>
          </div>
        </header>

        {/* KPI strip */}
        <KPIStrip strategy={strategy} />

        {/* Chart 1 — full width */}
        <div className="mb-8">
          <ChartCard title="Strategy KPI Comparison" icon={BarChart2}>
            <div className="h-[340px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={strategyData} barGap={4} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                  <XAxis dataKey="strategy" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} />
                  <Tooltip content={<Tip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                  <Legend wrapperStyle={{ fontSize: '11px', fontWeight: 'bold', paddingTop: '16px' }} />
                  <Bar dataKey="energy"  name="Energy (kWh)" fill="#f59e0b" radius={[6,6,0,0]} />
                  <Bar dataKey="co2"    name="CO₂ (kg)"    fill="#10b981" radius={[6,6,0,0]} />
                  <Bar dataKey="savings" name="Savings ($)"  fill="#06b6d4" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>

        {/* Charts 3 & 4 — side by side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Chart 3 — Radar */}
          <ChartCard title="KPI Performance Radar" icon={Target}>
            <div className="h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="75%">
                  <PolarGrid stroke="#ffffff12" />
                  <PolarAngleAxis dataKey="kpi" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 700 }} />
                  <Radar name="Current" dataKey="current" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.18} strokeWidth={2} />
                  <Radar name="Target"  dataKey="target"  stroke="#a855f7" fill="#a855f7" fillOpacity={0.1}  strokeWidth={2} strokeDasharray="4 2" />
                  <Legend wrapperStyle={{ fontSize: '11px', fontWeight: 'bold', paddingTop: '8px' }} />
                  <Tooltip content={<Tip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          {/* Chart 4 — Projection */}
          <ChartCard title="30-Day Savings Projection" icon={TrendingUp}>
            <div className="h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={projectionData}>
                  <defs>
                    <linearGradient id="optG"  x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="baseG" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#64748b" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#64748b" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                  <XAxis dataKey="day" stroke="#64748b" fontSize={11} interval={4} />
                  <YAxis stroke="#64748b" fontSize={11} tickFormatter={v => `$${v}`} />
                  <Tooltip content={<Tip />} />
                  <Legend wrapperStyle={{ fontSize: '11px', fontWeight: 'bold', paddingTop: '16px' }} />
                  <Area type="monotone" dataKey="baseline"  name="Baseline ($)"  stroke="#64748b" fill="url(#baseG)" strokeWidth={1.5} dot={false} />
                  <Area type="monotone" dataKey="optimized" name="Optimized ($)" stroke="#06b6d4" fill="url(#optG)"  strokeWidth={2.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

        </div>
      </div>
      <Chatbot />
    </div>
  );
};

export default Analytics;
