import React, { useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Cell
} from 'recharts';
import {
  Cpu, BarChart2, Search, Sliders, Zap, AlertTriangle, Activity
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ParticlesBackground from '../components/ParticlesBackground';
import Chatbot from '../components/Chatbot';
import { runSweep } from '../services/optibatchApi';

const SweepExplorer = () => {
  const navigate = useNavigate();
  
  const [temperatureRange, setTemperatureRange] = useState([135, 155]);
  const [pressureRange, setPressureRange] = useState([1.8, 2.5]);
  const [steps, setSteps] = useState(20);
  
  const [loading, setLoading] = useState(false);
  const [sweepResults, setSweepResults] = useState([]);
  const [bestConfig, setBestConfig] = useState(null);
  const [isHeuristic, setIsHeuristic] = useState(false);
  const [totalSims, setTotalSims] = useState(0);

  const handleRunSweep = async () => {
    setLoading(true);
    setSweepResults([]);
    setBestConfig(null);

    try {
      const data = await runSweep({
        temperature_range: temperatureRange,
        pressure_range: pressureRange,
        steps
      });

      if (data && data.best_simulated_batches) {
        setSweepResults(data.best_simulated_batches);
        setIsHeuristic(!!data.is_heuristic_fallback);
        setTotalSims(data.total_simulations || data.best_simulated_batches.length);
        
        // Compute best configuration directly from best_simulated_batches
        if (data.best_simulated_batches.length > 0) {
          const sorted = [...data.best_simulated_batches].sort((a, b) => {
            const yieldA = a.yield || 0;
            const yieldB = b.yield || 0;
            if (yieldA !== yieldB) return yieldB - yieldA;
            const energyA = a.energy_consumption || a.energy || 0;
            const energyB = b.energy_consumption || b.energy || 0;
            return energyA - energyB;
          });
          setBestConfig(sorted[0]);
        }
      }
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  // Convert for heatmap/scatter
  const chartData = sweepResults.map(r => ({
    temperature: r.temperature,
    pressure: r.pressure,
    yield: r.yield,
    energy: r.energy_consumption || r.energy || 150
  }));

  return (
    <div className="bg-[#050505] min-h-screen text-slate-200 font-sans selection:bg-cyan-500/30 relative">
      <ParticlesBackground />

      <div className="max-w-[1600px] mx-auto p-4 md:p-8 pt-8 relative z-10">
        {/* ── Header ──────────────────────────────────────────────── */}
        <header className="flex justify-between items-center mb-8 pb-6 border-b border-white/10">
          <div className="flex items-center gap-5">
            <div className="relative group cursor-pointer" onClick={() => navigate('/dashboard')}>
              <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-40 group-hover:opacity-60 transition duration-500" />
              <div className="relative bg-gradient-to-br from-gray-900 to-black border border-white/10 p-3 rounded-2xl">
                <Cpu className="text-cyan-400" size={32} />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-black tracking-tighter text-white">
                OPTI<span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">BATCH</span>{' '}
                <span className="text-lg font-light text-slate-400">v3.0</span>
              </h1>
              <p className="text-[11px] text-slate-500 uppercase tracking-[0.3em] font-bold mt-1">Parameter Sweep Explorer</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-3">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20 hover:scale-105 transition-all text-xs font-black tracking-widest uppercase"
            >
              <Activity size={14} /> Dashboard
            </button>
            <button
              onClick={() => navigate('/analytics')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20 hover:scale-105 transition-all text-xs font-black tracking-widest uppercase"
            >
              <BarChart2 size={14} /> Analytics
            </button>
          </div>
        </header>

        {/* Heuristic warning for sweep */}
        {isHeuristic && sweepResults.length > 0 && (
          <div className="mb-6 p-4 rounded-2xl border border-amber-500/30 bg-amber-500/5 flex items-center gap-4">
            <AlertTriangle size={20} className="text-amber-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-black text-amber-400 uppercase tracking-widest mb-0.5">Simulated Prediction Mode</p>
              <p className="text-[11px] text-slate-400">Sweep results are generated using heuristic equations, not trained ML models. Use for exploration only.</p>
            </div>
            <span className="text-xs font-bold text-slate-500 flex-shrink-0">{totalSims} sims</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* ── Control Panel ───────────────────────────────── */}
          <div className="glass-panel p-8 rounded-3xl flex flex-col">
            <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest mb-8 flex items-center gap-3">
              <Sliders size={18} className="text-cyan-400" /> Sweep Configuration
            </h3>
            
            <div className="space-y-6 flex-1">
              <div>
                <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Temperature Range</label>
                <div className="flex items-center gap-3">
                  <input type="number" value={temperatureRange[0]} onChange={e => setTemperatureRange([Number(e.target.value), temperatureRange[1]])} className="w-1/2 bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-cyan-500/50" />
                  <span className="text-slate-500">-</span>
                  <input type="number" value={temperatureRange[1]} onChange={e => setTemperatureRange([temperatureRange[0], Number(e.target.value)])} className="w-1/2 bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-cyan-500/50" />
                </div>
              </div>

              <div>
                <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Pressure Range</label>
                <div className="flex items-center gap-3">
                  <input type="number" step="0.1" value={pressureRange[0]} onChange={e => setPressureRange([Number(e.target.value), pressureRange[1]])} className="w-1/2 bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-cyan-500/50" />
                  <span className="text-slate-500">-</span>
                  <input type="number" step="0.1" value={pressureRange[1]} onChange={e => setPressureRange([pressureRange[0], Number(e.target.value)])} className="w-1/2 bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-cyan-500/50" />
                </div>
              </div>

              <div>
                <label className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2">Simulations (Steps)</label>
                <input type="number" value={steps} onChange={e => setSteps(Number(e.target.value))} className="w-full bg-white/[0.03] border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:border-cyan-500/50" />
              </div>
            </div>

            <button 
              onClick={handleRunSweep} 
              disabled={loading}
              className="mt-8 w-full py-4 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-black uppercase tracking-widest hover:bg-cyan-500/20 transition-colors disabled:opacity-50 flex justify-center items-center gap-3"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                  AI Exploring Parameter Space...
                </>
              ) : (
                <>
                  <Search size={18} /> Run Parameter Sweep
                </>
              )}
            </button>
          </div>

          {/* ── Results Visualization ───────────────────────────────── */}
          <div className="lg:col-span-2 flex flex-col gap-8">
            <div className="glass-panel p-8 rounded-3xl relative overflow-hidden flex-1">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
              <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3 mb-6">
                <BarChart2 size={18} className="text-cyan-400" /> Energy vs Yield Scatter
              </h3>

              {!loading && sweepResults.length === 0 && (
                <div className="h-[280px] w-full flex items-center justify-center text-slate-600 font-bold tracking-widest uppercase text-xs">
                  Run simulation to visualize parameter space
                </div>
              )}

              {sweepResults.length > 0 && (
                <div className="h-[280px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                      <XAxis type="number" dataKey="energy" name="Energy" stroke="#64748b" tickMargin={10} domain={['auto', 'auto']} />
                      <YAxis type="number" dataKey="yield" name="Yield" stroke="#64748b" tickMargin={10} domain={['auto', 'auto']} />
                      <RechartsTooltip 
                        cursor={{ strokeDasharray: '3 3' }} 
                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)' }}
                        itemStyle={{ color: '#e2e8f0', fontWeight: 'bold' }}
                      />
                      <Scatter name="Simulations" data={chartData} fill="#06b6d4">
                        {chartData.map((entry, index) => {
                           const isBest = bestConfig && entry.temperature === bestConfig.temperature && entry.pressure === bestConfig.pressure;
                           return <Cell key={`cell-${index}`} fill={isBest ? '#10b981' : '#06b6d4'} />;
                        })}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {bestConfig && (
              <div className="glass-panel p-8 rounded-3xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-emerald-500/50 to-transparent" />
                <h3 className="text-sm font-black text-emerald-400 uppercase tracking-widest flex items-center gap-3 mb-6">
                  <Zap size={18} /> Optimal Zone Identified
                </h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                    <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Temperature</p>
                    <p className="text-xl font-black text-white">{bestConfig.temperature?.toFixed(2)} °C</p>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                    <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Pressure</p>
                    <p className="text-xl font-black text-white">{bestConfig.pressure?.toFixed(2)} atm</p>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                    <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Assoc. Yield</p>
                    <p className="text-xl font-black text-emerald-400">{bestConfig.yield?.toFixed(2)}%</p>
                  </div>
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
                    <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Est. Energy</p>
                    <p className="text-xl font-black text-white">{(bestConfig.energy_consumption || bestConfig.energy || 150).toFixed(1)} kWh</p>
                  </div>
                </div>
                {bestConfig.prediction_confidence !== undefined && (
                  <div className="mt-4 flex items-center gap-3">
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Prediction Confidence</span>
                    <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-700 ${bestConfig.prediction_confidence >= 0.7 ? 'bg-emerald-400' : bestConfig.prediction_confidence >= 0.4 ? 'bg-amber-400' : 'bg-red-400'}`}
                        style={{ width: `${Math.round(bestConfig.prediction_confidence * 100)}%` }} 
                      />
                    </div>
                    <span className={`text-xs font-black ${bestConfig.prediction_confidence >= 0.7 ? 'text-emerald-400' : bestConfig.prediction_confidence >= 0.4 ? 'text-amber-400' : 'text-red-400'}`}>
                      {Math.round(bestConfig.prediction_confidence * 100)}%
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

      </div>

      <Chatbot />
    </div>
  );
};

export default SweepExplorer;
