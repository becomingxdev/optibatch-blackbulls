import React, { useState, useEffect, useRef } from 'react';
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, ComposedChart, ReferenceLine, Legend,
  BarChart, Bar, Cell
} from 'recharts';
import {
  Activity, Zap, Leaf, AlertTriangle, Info,
  Settings, Cpu, ShieldCheck, TrendingDown, BarChart2,
  ClipboardList, BrainCircuit, CheckCircle2, AlertOctagon, ChevronUp, ChevronDown,
  Search, Eye
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ParticlesBackground from '../components/ParticlesBackground';
import Chatbot from '../components/Chatbot';
import { monitorBatch, optimizeBatch } from '../services/optibatchApi';

/* ─── Reusable KPI Card ─────────────────────────────────────────────── */
const GlassCard = ({ title, value, icon, color, pulse, subtitle }) => {
  const styles = {
    cyan:    { text: 'text-cyan-400',    bg: 'bg-cyan-400/10',    border: 'border-cyan-400/20',    shadow: 'shadow-[0_0_15px_rgba(34,211,238,0.1)]' },
    amber:   { text: 'text-amber-400',   bg: 'bg-amber-400/10',   border: 'border-amber-400/20',   shadow: 'shadow-[0_0_15px_rgba(251,191,36,0.1)]' },
    emerald: { text: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/20', shadow: 'shadow-[0_0_15px_rgba(16,185,129,0.1)]' },
    red:     { text: 'text-red-400',     bg: 'bg-red-400/10',     border: 'border-red-500/30',     shadow: 'shadow-[0_0_20px_rgba(239,68,68,0.2)]' },
  };
  const c = styles[color];
  return (
    <div className={`glass-panel p-6 rounded-3xl transition-all duration-300 hover:translate-y-[-4px] ${c.shadow} ${pulse ? 'animate-pulse ring-1 ring-red-500/50' : ''}`}>
      <div className="flex justify-between items-start mb-6">
        <div className={`p-3 rounded-2xl ${c.bg} ${c.border} border backdrop-blur-md`}>
          {React.cloneElement(icon, { className: c.text })}
        </div>
        <div className="text-[9px] font-black px-2 py-1 rounded-full bg-white/5 text-slate-500 uppercase tracking-widest border border-white/5">Active</div>
      </div>
      <div>
        <p className="text-[11px] text-slate-400 uppercase font-black tracking-widest mb-1">{title}</p>
        <p className="text-3xl font-black text-white tracking-tighter">{value}</p>
        {subtitle && <p className="text-[10px] text-slate-500 mt-1">{subtitle}</p>}
      </div>
    </div>
  );
};

/* ─── Heuristic Fallback Warning Banner ────────────────────────────── */
const HeuristicBanner = ({ isHeuristic, confidence }) => {
  if (!isHeuristic) return null;
  return (
    <div className="mb-6 p-4 rounded-2xl border border-amber-500/30 bg-amber-500/5 flex items-center gap-4">
      <div className="flex-shrink-0 p-2 rounded-xl bg-amber-500/10 border border-amber-500/20">
        <AlertTriangle size={20} className="text-amber-400" />
      </div>
      <div className="flex-1">
        <p className="text-xs font-black text-amber-400 uppercase tracking-widest mb-0.5">Simulated Prediction Mode</p>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          No trained ML models found. Predictions are generated using deterministic heuristic equations.
          Results are approximate and should not be used for critical process decisions.
        </p>
      </div>
      <div className="flex-shrink-0 text-right">
        <p className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Confidence</p>
        <p className="text-lg font-black text-amber-400">{((confidence || 0) * 100).toFixed(0)}%</p>
      </div>
    </div>
  );
};

/* ─── Prediction Confidence Bar ──────────────────────────────────── */
const ConfidenceBar = ({ confidence, label }) => {
  const pct = Math.round((confidence || 0) * 100);
  const color = pct >= 70 ? 'bg-emerald-400' : pct >= 40 ? 'bg-amber-400' : 'bg-red-400';
  const textColor = pct >= 70 ? 'text-emerald-400' : pct >= 40 ? 'text-amber-400' : 'text-red-400';
  return (
    <div className="flex items-center gap-3">
      {label && <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider w-20 flex-shrink-0">{label}</span>}
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs font-black ${textColor} w-10 text-right`}>{pct}%</span>
    </div>
  );
};

/* ─── Adaptive Suggestions ── Modified to accept backend alerts ──────── */
const AdaptiveSuggestions = ({ data, strategy, alertMessage, batchStatus, optimizationSuggestion, optimizationExplanation }) => {
  if (!data || data.length === 0) return null;

  const recent = data.slice(-5);
  const avgTemp = recent.reduce((s, d) => s + d.Temperature_C, 0) / recent.length;
  const avgMid  = recent.reduce((s, d) => s + ((d.Target_Upper + d.Target_Lower) / 2), 0) / recent.length;
  const deviation = avgTemp - avgMid;
  const absDev = Math.abs(deviation);
  const trend = data.length > 3
    ? data[data.length - 1].Temperature_C - data[data.length - 4].Temperature_C
    : 0;

  const suggestions = [];

  if (alertMessage) {
    const level = batchStatus?.toLowerCase().includes('critical') ? 'critical' : 'warning';
    suggestions.push({
      level: level,
      icon: AlertOctagon,
      title: `System Alert: ${batchStatus || 'Deviation'}`,
      detail: alertMessage,
    });
  }

  if (optimizationSuggestion && Object.keys(optimizationSuggestion).length > 0) {
    suggestions.push({
      level: 'ok',
      icon: Settings,
      title: 'AI Optimization Recommendation',
      detail: `Optimal configuration: ${Object.entries(optimizationSuggestion).map(([k, v]) => `${k.replace('_', ' ')}: ${v}`).join(', ')}`,
    });
  }

  if (optimizationExplanation) {
    suggestions.push({
      level: 'ok',
      icon: Info,
      title: 'Optimization Rationale',
      detail: optimizationExplanation,
    });
  }

  if (absDev > 1.5) {
    suggestions.push({
      level: 'critical',
      icon: AlertOctagon,
      title: deviation > 0 ? 'Temperature Above Target' : 'Temperature Below Target',
      detail: `Running ${absDev.toFixed(2)}°C ${deviation > 0 ? 'above' : 'below'} golden batch midpoint. ${deviation > 0 ? 'Reduce heater output by ~8%.' : 'Increase heater output by ~6%.'}`,
    });
  } else if (absDev > 0.5) {
    suggestions.push({
      level: 'warning',
      icon: AlertTriangle,
      title: 'Minor Deviation Detected',
      detail: `Temperature is ${absDev.toFixed(2)}°C from optimal. Consider adjusting ±3% to stay within golden envelope.`,
    });
  } else {
    suggestions.push({
      level: 'ok',
      icon: CheckCircle2,
      title: 'Within Golden Envelope',
      detail: `Temperature is tracking the golden batch signature closely (Δ${absDev.toFixed(2)}°C). No intervention needed.`,
    });
  }

  if (trend > 0.8) {
    suggestions.push({
      level: 'warning',
      icon: ChevronUp,
      title: 'Rising Trend Detected',
      detail: `Temperature rising +${trend.toFixed(2)}°C over last 3 intervals. Pre-emptively lower heater setpoint by 5%.`,
    });
  } else if (trend < -0.8) {
    suggestions.push({
      level: 'warning',
      icon: ChevronDown,
      title: 'Falling Trend Detected',
      detail: `Temperature dropping ${Math.abs(trend).toFixed(2)}°C over last 3 intervals. Increase heater to compensate.`,
    });
  }

  if (strategy < 40) {
    suggestions.push({
      level: 'critical',
      icon: AlertOctagon,
      title: 'Low Efficiency Strategy',
      detail: `Strategy at ${strategy}% wastes significant energy. AI recommends minimum 60% for optimal throughput-to-cost ratio.`,
    });
  } else if (strategy >= 85) {
    suggestions.push({
      level: 'ok',
      icon: CheckCircle2,
      title: 'High Performance Mode',
      detail: `Strategy at ${strategy}% — maximum savings and efficiency. Monitor for thermal overshoot risk.`,
    });
  }

  const colors = {
    critical: { bg: 'bg-red-500/5',     border: 'border-red-500/20',     text: 'text-red-400',     dot: 'bg-red-400' },
    warning:  { bg: 'bg-amber-500/5',   border: 'border-amber-500/20',   text: 'text-amber-400',   dot: 'bg-amber-400' },
    ok:       { bg: 'bg-emerald-500/5', border: 'border-emerald-500/20', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  };

  return (
    <div className="glass-panel p-6 rounded-3xl relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />
      <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3 mb-5">
        <BrainCircuit size={17} className="text-violet-400" /> AI Adaptive Corrections
      </h3>
      <div className="flex flex-col gap-3">
        {suggestions.map((s, i) => {
          const Icon = s.icon;
          const c = colors[s.level];
          return (
            <div key={i} className={`flex gap-3 p-4 rounded-xl border ${c.bg} ${c.border}`}>
              <div className={`flex-shrink-0 mt-0.5 ${c.text}`}><Icon size={16} /></div>
              <div>
                <p className={`text-xs font-black uppercase tracking-wider ${c.text} mb-0.5`}>{s.title}</p>
                <p className="text-[11px] text-slate-400 leading-relaxed">{s.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ─── Operator Decision Log ─────────────────────────────────────────── */
const OperatorLog = ({ log }) => (
  <div className="glass-panel p-6 rounded-3xl relative overflow-hidden">
    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-amber-500/40 to-transparent" />
    <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3 mb-5">
      <ClipboardList size={17} className="text-amber-400" /> Operator Decision Log
    </h3>
    {log.length === 0 ? (
      <p className="text-xs text-slate-500 italic">No overrides yet — move the strategy slider to log decisions.</p>
    ) : (
      <div className="flex flex-col gap-2 max-h-[200px] overflow-y-auto pr-1">
        {[...log].reverse().map((entry, i) => (
          <div key={i} className="flex items-center justify-between py-2.5 px-3 rounded-xl bg-white/[0.03] border border-white/5 text-xs">
            <div className="flex items-center gap-3">
              <div className="w-1.5 h-1.5 rounded-full bg-amber-400 flex-shrink-0" />
              <span className="text-slate-300 font-bold">Strategy → <span className="text-amber-300">{entry.strategy}%</span></span>
            </div>
            <span className="text-slate-600 font-mono text-[10px]">{entry.time}</span>
          </div>
        ))}
      </div>
    )}
  </div>
);

/* ─── AI Optimization Impact Panel ──────────────────────────────────── */
const OptimizationComparisonPanel = ({ comparison, confidence }) => {
  if (!comparison) return null;

  const calculateImprovement = (current, optimized) => {
    if (!current || !optimized) return 0;
    return (((optimized - current) / current) * 100);
  };

  const data = [
    {
      name: 'Energy',
      Current: comparison.energy.current,
      Optimized: comparison.energy.optimized,
      improvement: calculateImprovement(comparison.energy.current, comparison.energy.optimized)
    },
    {
      name: 'Cost',
      Current: comparison.cost.current,
      Optimized: comparison.cost.optimized,
      improvement: calculateImprovement(comparison.cost.current, comparison.cost.optimized)
    },
    {
      name: 'Yield',
      Current: comparison.yield.current,
      Optimized: comparison.yield.optimized,
      improvement: calculateImprovement(comparison.yield.current, comparison.yield.optimized)
    }
  ];

  return (
    <div className="glass-panel p-6 rounded-3xl relative overflow-hidden mb-8">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent" />
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3">
          <BarChart2 size={18} className="text-cyan-400" /> AI Optimization Impact
        </h3>
        {confidence !== undefined && confidence !== null && (
          <div className="flex items-center gap-2">
            <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider">Optimization Confidence</span>
            <ConfidenceBar confidence={confidence} />
          </div>
        )}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        <div className="bg-white/[0.03] border border-white/5 rounded-2xl overflow-hidden w-full">
          <table className="w-full text-xs text-left">
            <thead className="bg-white/[0.02] border-b border-white/5 uppercase font-black text-slate-500 tracking-wider">
              <tr>
                <th className="p-4">Metric</th>
                <th className="p-4 text-right">Current</th>
                <th className="p-4 text-right">Optimized</th>
                <th className="p-4 text-right">Improvement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {data.map((row, i) => {
                const better = row.name === 'Yield' ? row.improvement > 0 : row.improvement < 0;
                return (
                  <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                    <td className="p-4 font-bold text-slate-300">{row.name}</td>
                    <td className="p-4 text-right text-slate-400">{row.Current.toFixed(1)}</td>
                    <td className="p-4 text-right text-cyan-400 font-bold">{row.Optimized.toFixed(1)}</td>
                    <td className={`p-4 text-right font-black ${better ? 'text-emerald-400' : 'text-red-400'}`}>
                      {row.improvement > 0 ? '+' : ''}{row.improvement.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        
        <div className="h-[220px] w-full mt-4 lg:mt-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickMargin={10} />
              <YAxis stroke="#64748b" fontSize={11} tickMargin={10} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)' }}
                itemStyle={{ color: '#e2e8f0', fontWeight: 'bold' }}
                cursor={{ fill: '#ffffff05' }}
              />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '11px', fontWeight: 'bold', color: '#94a3b8' }} />
              <Bar dataKey="Current" fill="#475569" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Optimized" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

/* ─── Historical Context Panel ──────────────────────────────────────── */
const HistoricalContextPanel = ({ historicalMetrics, currentMetrics }) => {
  if (!historicalMetrics || !currentMetrics) return null;

  const metrics = [
    {
      name: 'Yield (%)',
      historical: historicalMetrics.yield || [],
      current: currentMetrics.yield_percentage || 0,
      color: '#10b981',
    },
    {
      name: 'Energy (kWh)',
      historical: historicalMetrics.energy || [],
      current: currentMetrics.energy_consumption || 0,
      color: '#f59e0b',
    },
    {
      name: 'Quality',
      historical: historicalMetrics.quality || [],
      current: currentMetrics.quality || 0,
      color: '#06b6d4',
    },
  ];

  return (
    <div className="glass-panel p-6 rounded-3xl relative overflow-hidden mb-8">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />
      <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3 mb-6">
        <Eye size={18} className="text-violet-400" /> Historical Context vs Current Prediction
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metrics.map((m, idx) => {
          if (m.historical.length === 0 && !m.current) return null;
          const avg = m.historical.length > 0
            ? m.historical.reduce((a, b) => a + b, 0) / m.historical.length
            : 0;
          const min = m.historical.length > 0 ? Math.min(...m.historical) : 0;
          const max = m.historical.length > 0 ? Math.max(...m.historical) : 0;
          const deviation = avg !== 0 ? ((m.current - avg) / avg * 100).toFixed(1) : '0.0';
          const isGood = m.name.includes('Yield') || m.name.includes('Quality')
            ? m.current >= avg
            : m.current <= avg;

          return (
            <div key={idx} className="p-4 rounded-2xl bg-white/[0.02] border border-white/5">
              <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-3">{m.name}</p>
              <div className="flex items-end gap-2 mb-3">
                <span className="text-2xl font-black text-white">{m.current.toFixed(1)}</span>
                <span className={`text-xs font-bold ${isGood ? 'text-emerald-400' : 'text-red-400'} mb-1`}>
                  {deviation > 0 ? '+' : ''}{deviation}%
                </span>
              </div>
              <div className="space-y-1.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-slate-500">Historical Avg</span>
                  <span className="text-slate-400 font-bold">{avg.toFixed(1)}</span>
                </div>
                <div className="flex justify-between text-[10px]">
                  <span className="text-slate-500">Range</span>
                  <span className="text-slate-400 font-bold">{min.toFixed(1)} – {max.toFixed(1)}</span>
                </div>
                {/* Mini bar showing current vs historical range */}
                <div className="relative h-3 bg-white/5 rounded-full overflow-hidden mt-2">
                  {/* Historical range fill */}
                  <div className="absolute h-full bg-white/10 rounded-full" style={{ left: '10%', width: '80%' }} />
                  {/* Current position indicator */}
                  <div
                    className="absolute top-0 h-full w-1.5 rounded-full"
                    style={{
                      backgroundColor: m.color,
                      left: `${Math.max(5, Math.min(95, ((m.current - min) / (max - min || 1)) * 80 + 10))}%`,
                      boxShadow: `0 0 6px ${m.color}`,
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ─── Dashboard Page ────────────────────────────────────────────────── */
const Dashboard = () => {
  const navigate = useNavigate();
  const [strategy, setStrategy]       = useState(() => {
    return Number(localStorage.getItem('optibatch_strategy')) || 70;
  });
  const [temperature, setTemperature] = useState(150.0);
  const [pressure, setPressure]       = useState(2.5);
  const [hold_time, setHoldTime]      = useState(30.0);
  const [catalyst_ratio, setCatalystRatio] = useState(1.2);

  const [backendData, setBackendData] = useState([]);
  const [loading, setLoading]         = useState(true);
  const [optimizing, setOptimizing]   = useState(false);
  
  const [predictedMetrics, setPredictedMetrics] = useState({});
  const [closestSignature, setClosestSignature] = useState('');
  const [driftScore, setDriftScore] = useState(0.0);
  const [batchStatus, setBatchStatus] = useState('');
  const [alertMessage, setAlertMessage] = useState('');
  const [optimizationSuggestion, setOptimizationSuggestion] = useState({});
  
  // Trust-centric state
  const [isHeuristicFallback, setIsHeuristicFallback] = useState(false);
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const [optimizationConfidence, setOptimizationConfidence] = useState(null);
  const [optimizationExplanation, setOptimizationExplanation] = useState('');
  const [historicalMetrics, setHistoricalMetrics] = useState(null);

  const [stats, setStats]             = useState({ energy: '0.0 kWh', co2: '0.00 kg', savings: '$0.00' });
  const [opLog, setOpLog]             = useState([]);
  const [optimizationComparison, setOptimizationComparison] = useState(null);
  const prevStrategy                  = useRef(70);

  /* API fetch */
  useEffect(() => {
    setLoading(true);

    const batchParameters = {
      temperature,
      pressure,
      hold_time,
      catalyst_ratio,
      strategy
    };

    monitorBatch(batchParameters)
      .then(data => {
        if (!data) return;

        if (data.predicted_metrics) setPredictedMetrics(data.predicted_metrics);
        if (data.closest_signature) setClosestSignature(data.closest_signature);
        if (data.drift_score !== undefined) setDriftScore(data.drift_score);
        if (data.batch_status) setBatchStatus(data.batch_status);
        if (data.alert_message) setAlertMessage(data.alert_message);
        if (data.optimization_suggestion) setOptimizationSuggestion(data.optimization_suggestion);

        // Trust metadata
        setIsHeuristicFallback(!!data.is_heuristic_fallback);
        setPredictionConfidence(data.prediction_confidence || 0);
        if (data.historical_metrics) setHistoricalMetrics(data.historical_metrics);

        const chartDataRaw = data.telemetryData || data.chart_data || [];
        if (chartDataRaw && chartDataRaw.length > 0) {
          const enriched = chartDataRaw.map(d => ({
            ...d,
            Golden_Batch: parseFloat(((d.Target_Upper + d.Target_Lower) / 2).toFixed(3)),
          }));
          setBackendData(enriched);
        }
        
        if (data.predicted_metrics) {
           setStats({
             energy: data.predicted_metrics.energy_consumption ? `${data.predicted_metrics.energy_consumption.toFixed(1)} kWh` : '0.0 kWh',
             co2: data.predicted_metrics.production_cost ? `$${data.predicted_metrics.production_cost.toFixed(2)}` : '$0.00',
             savings: data.predicted_metrics.yield_percentage ? `${data.predicted_metrics.yield_percentage.toFixed(1)}% Yield` : '0.0%'
           });
        }
        setLoading(false);
      })
      .catch(err => { 
        console.error('Monitoring Error:', err); 
        setLoading(false); 
      });
  }, [strategy, temperature, pressure, hold_time, catalyst_ratio]);

  /* Log operator overrides when strategy changes */
  const handleStrategyChange = async (e) => {
    const val = Number(e.target.value);
    setStrategy(val);
    localStorage.setItem('optibatch_strategy', val);
    
    // Only log and optimize when slider stops on a new value
    if (val !== prevStrategy.current) {
      prevStrategy.current = val;
      const now = new Date();
      const time = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      setOpLog(prev => [...prev, { strategy: val, time }]);
      
      setOptimizing(true);
      try {
        const optimization_goal = val < 40 ? "minimize_energy" : val <= 70 ? "balanced" : "maximize_yield";
        
        const currentMetrics = predictedMetrics;
        
        const result = await optimizeBatch({
          temperature,
          pressure,
          hold_time,
          catalyst_ratio,
          strategy: val,
          optimization_goal
        }, currentMetrics);

        if (result) {
          // Trust metadata from optimization
          if (result.optimization_confidence !== undefined) setOptimizationConfidence(result.optimization_confidence);
          if (result.optimization_explanation) setOptimizationExplanation(result.optimization_explanation);

          if (result.predicted_metrics && currentMetrics && Object.keys(currentMetrics).length > 0) {
            const optimizedMetrics = result.predicted_metrics;
            setOptimizationComparison({
              energy: {
                current: currentMetrics.energy_consumption || 0,
                optimized: optimizedMetrics.energy_consumption || 0
              },
              cost: {
                current: currentMetrics.production_cost || 0,
                optimized: optimizedMetrics.production_cost || 0
              },
              yield: {
                current: currentMetrics.yield_percentage || 0,
                optimized: optimizedMetrics.yield_percentage || 0
              }
            });
          }

          if (result.parameter_recommendations || result.optimal_parameters) {
            setOptimizationSuggestion(result.parameter_recommendations || result.optimal_parameters);
            
            // Apply optimized parameters visually to shift the monitoring graph and context bounds!
            if (result.optimal_parameters) {
              if (result.optimal_parameters.temperature) setTemperature(Number(result.optimal_parameters.temperature));
              if (result.optimal_parameters.pressure) setPressure(Number(result.optimal_parameters.pressure));
              if (result.optimal_parameters.hold_time) setHoldTime(Number(result.optimal_parameters.hold_time));
              if (result.optimal_parameters.catalyst_ratio) setCatalystRatio(Number(result.optimal_parameters.catalyst_ratio));
            }
          }
          
          if (result.predicted_metrics) {
            setStats(prev => ({
              ...prev,
              energy: result.predicted_metrics.energy_consumption ? `${result.predicted_metrics.energy_consumption.toFixed(1)} kWh` : prev.energy,
              co2: result.predicted_metrics.production_cost ? `$${result.predicted_metrics.production_cost.toFixed(2)}` : prev.co2,
            }));
          }
          
          if (result.expected_improvement) {
             setStats(prev => ({ ...prev, savings: result.expected_improvement }));
          }
        }
      } catch (err) {
        console.error("Optimization error:", err);
      }
      setOptimizing(false);
    }
  };

  const isWarning = strategy < 30;

  return (
    <div className="bg-[#050505] min-h-screen text-slate-200 font-sans selection:bg-cyan-500/30 relative">
      <ParticlesBackground />

      <div className="max-w-[1600px] mx-auto p-4 md:p-8 pt-8 relative z-10">

        {/* ── Header ──────────────────────────────────────────────── */}
        <header className="flex justify-between items-center mb-8 pb-6 border-b border-white/10">
          <div className="flex items-center gap-5">
            <div className="relative group cursor-pointer" onClick={() => navigate('/')}>
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
              <p className="text-[11px] text-slate-500 uppercase tracking-[0.3em] font-bold mt-1">Industrial Optimization Engine</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-3">
            <button
              onClick={() => navigate('/analytics')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20 hover:scale-105 transition-all text-xs font-black tracking-widest uppercase"
            >
              <BarChart2 size={14} /> Analytics
            </button>
            <button
              onClick={() => navigate('/sweep-explorer')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20 hover:scale-105 transition-all text-xs font-black tracking-widest uppercase"
            >
              <Search size={14} /> Sweep Explorer
            </button>
            <div className="flex px-5 py-2.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 items-center gap-3 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
              <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
              <span className="text-xs font-black text-emerald-400 tracking-widest uppercase">Telemetry Live</span>
            </div>
          </div>
        </header>

        {/* ── Heuristic Warning Banner ─────────────────────────────── */}
        <HeuristicBanner isHeuristic={isHeuristicFallback} confidence={predictionConfidence} />

        {/* ── Prediction Confidence Strip ──────────────────────────── */}
        {!loading && (
          <div className="glass-panel p-4 rounded-2xl mb-6 flex flex-col md:flex-row items-center gap-4">
            <div className="flex items-center gap-2 flex-shrink-0">
              <ShieldCheck size={16} className={predictionConfidence >= 0.7 ? 'text-emerald-400' : predictionConfidence >= 0.4 ? 'text-amber-400' : 'text-red-400'} />
              <span className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Prediction Reliability</span>
            </div>
            <div className="flex-1 w-full">
              <ConfidenceBar confidence={predictionConfidence} />
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Drift Score</span>
              <span className={`text-xs font-black ${driftScore > 0.5 ? 'text-red-400' : driftScore > 0.2 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {(driftScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}

        {/* ── Stat Cards ──────────────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <GlassCard title="Energy Load"   value={stats.energy} icon={<Zap size={22} />}          color="amber"   />
          <GlassCard title="Production Cost" value={stats.co2}    icon={<Leaf size={22} />}         color="emerald" />
          <GlassCard
            title="Batch Status"
            value={batchStatus ? batchStatus.toUpperCase() : 'OPTIMAL'}
            icon={batchStatus?.toLowerCase().includes('critical') || batchStatus?.toLowerCase().includes('warning') ? <AlertTriangle size={22} /> : <ShieldCheck size={22} />}
            color={batchStatus?.toLowerCase().includes('critical') ? 'red' : 'cyan'}
            pulse={batchStatus?.toLowerCase().includes('critical')}
          />
        </div>

        {/* ── Main Grid ── Chart + Control ────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">

          {/* ── Thermal Chart with Golden Batch ─────────────────── */}
          <div className="lg:col-span-2 glass-panel p-8 rounded-3xl relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest flex items-center gap-3">
                <Activity size={18} className="text-cyan-400" /> Thermal Signature Array
              </h3>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <div className="w-4 h-0.5 bg-amber-400 rounded" style={{ backgroundImage: 'repeating-linear-gradient(90deg,#fbbf24 0,#fbbf24 4px,transparent 4px,transparent 8px)' }} />
                  <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">Golden Batch</span>
                </div>
                <span className="text-xs font-mono text-slate-500">REFRESH: 100ms</span>
              </div>
            </div>
            {loading ? (
              <div className="h-[380px] w-full flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                  <div className="w-10 h-10 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm text-slate-500 uppercase tracking-widest font-bold">Fetching Telemetry...</span>
                </div>
              </div>
            ) : backendData.length === 0 ? (
               <div className="h-[380px] w-full flex items-center justify-center border-t border-white/5 mt-4">
                 <div className="flex flex-col items-center gap-4 opacity-50">
                    <Activity size={32} className="text-slate-500" />
                   <span className="text-sm text-slate-500 uppercase tracking-widest font-bold">Live Telemetry Stream Unavailable</span>
                   <span className="text-[10px] text-slate-600">Actual hardware connections required for time-series array</span>
                 </div>
               </div>
            ) : (
              <div className="h-[380px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={backendData}>
                    <defs>
                      <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#06b6d4" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" vertical={false} />
                    <XAxis dataKey="Time_Minutes" stroke="#64748b" fontSize={11} tickFormatter={v => `${v}m`} tickMargin={10} />
                    <YAxis domain={['auto', 'auto']} stroke="#64748b" fontSize={11} tickMargin={10} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)' }}
                      itemStyle={{ color: '#e2e8f0', fontWeight: 'bold' }}
                      cursor={{ stroke: '#ffffff12', strokeWidth: 1 }}
                    />
                    <Area  type="monotone" dataKey="Target_Upper" stroke="none" fill="url(#colorTemp)" name="Upper Bound" />
                    {/* Golden Batch — amber dashed ideal line */}
                    <Line  type="monotone" dataKey="Golden_Batch"  stroke="#fbbf24" strokeWidth={2} strokeDasharray="6 4" dot={false} name="Golden Batch" />
                    {/* Live temperature */}
                    <Line  type="monotone" dataKey="Temperature_C" stroke="#38bdf8" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: '#0ea5e9', stroke: '#fff' }} name="Live Temp (°C)" />
                    {/* Lower bound */}
                    <Line  type="monotone" dataKey="Target_Lower"  stroke="#ef4444" strokeWidth={1} strokeDasharray="4 4" dot={false} name="Lower Bound" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* ── Control + Savings ───────────────────────────────── */}
          <div className="glass-panel p-8 rounded-3xl flex flex-col justify-between">
            <div>
              <h3 className="text-sm font-black text-slate-300 uppercase tracking-widest mb-8 flex items-center gap-3">
                <Settings size={18} className="text-cyan-400" /> System Override
                {optimizing && <span className="ml-auto text-[10px] text-cyan-400 animate-pulse tracking-widest uppercase bg-cyan-500/10 px-2 py-1 rounded">AI Optimizing...</span>}
              </h3>
              <div className="space-y-6 mb-10 pt-4">
                <div className="flex items-center gap-4 w-full">
                  <span className="text-sm font-bold text-slate-400 w-24 flex-shrink-0">Efficiency</span>
                  <div className="flex-1 relative flex items-center">
                    <input
                      type="range"
                      className="custom-slider w-full"
                      value={strategy}
                      min="0"
                      max="100"
                      onChange={handleStrategyChange}
                      style={{ '--val': `${strategy}%` }}
                    />
                  </div>
                  <span className="text-sm font-bold text-slate-400 w-12 text-right flex-shrink-0">{strategy}%</span>
                </div>
              </div>
            </div>
            <div className="bg-gradient-to-br from-emerald-500/10 to-transparent border border-emerald-500/20 p-6 rounded-2xl relative overflow-hidden">
              <div className="absolute -right-4 -top-4 opacity-10">
                <TrendingDown size={100} className="text-emerald-500" />
              </div>
              <p className="text-[11px] text-emerald-400/80 uppercase font-black tracking-widest mb-2">Projected Total Savings</p>
              <div className="text-5xl font-black text-white tracking-tighter drop-shadow-lg">
                {stats.savings}
              </div>
            </div>
          </div>
        </div>

        {/* ── Bottom Row: AI Suggestions + Operator Log ────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <AdaptiveSuggestions data={backendData} strategy={Number(strategy)} alertMessage={alertMessage} batchStatus={batchStatus} optimizationSuggestion={optimizationSuggestion} optimizationExplanation={optimizationExplanation} />
          <OperatorLog log={opLog} />
        </div>

        {/* ── AI Optimization Impact panel ────────────── */}
        {optimizationComparison && <OptimizationComparisonPanel comparison={optimizationComparison} confidence={optimizationConfidence} />}

        {/* ── Historical Context Panel ────────────────── */}
        <HistoricalContextPanel historicalMetrics={historicalMetrics} currentMetrics={predictedMetrics} />

      </div>

      <Chatbot />
    </div>
  );
};

export default Dashboard;
