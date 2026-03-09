import React from 'react';
import { ArrowDown, Cpu, Activity, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Hero = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen w-full flex flex-col items-center justify-center px-6 overflow-hidden z-20 gap-8">
      
      {/* Decorative Floating Elements */}
      <div className="absolute top-1/4 left-1/4 animate-pulse opacity-20">
        <Cpu size={120} className="text-cyan-500" />
      </div>
      <div className="absolute bottom-1/3 right-1/4 animate-pulse opacity-10" style={{ animationDelay: '1s' }}>
        <Zap size={150} className="text-teal-400" />
      </div>

      {/* Main Content Panel */}
      <div className="p-12 md:p-20 rounded-[3rem] text-center max-w-4xl w-full relative group">

        <div className="inline-flex items-center gap-3 bg-cyan-500/10 border border-cyan-500/20 px-6 py-2 rounded-full mb-8">
          <Activity size={18} className="text-cyan-400" />
          <span className="text-sm font-bold text-cyan-300 uppercase tracking-widest">Next-Gen Industrial Control</span>
        </div>

        <h1 className="text-8xl md:text-[10rem] font-black text-transparent bg-clip-text bg-gradient-to-br from-white via-cyan-100 to-teal-500 tracking-tighter mb-6 drop-shadow-lg leading-none">
          OptiBatch
        </h1>
        
        <p className="text-xl md:text-2xl text-slate-300 max-w-2xl mx-auto font-light leading-relaxed mb-12">
          Intelligent telemetry and predictive AI optimization for modern industrial workflows. Empowered by <span className="font-bold text-white">Team BlackBull</span>.
        </p>

        {/* Navigate to /dashboard */}
        <button 
          onClick={() => navigate('/dashboard')}
          className="group relative inline-flex items-center justify-center gap-4 px-10 py-5 bg-gradient-to-r from-cyan-600 to-teal-600 rounded-full font-black text-white text-lg tracking-widest uppercase overflow-hidden shadow-[0_0_40px_rgba(45,212,191,0.4)] hover:shadow-[0_0_60px_rgba(45,212,191,0.6)] transition-all duration-300 hover:scale-105"
        >
          <span>Launch Telemetry</span>
          <ArrowDown size={24} className="group-hover:animate-bounce" />
          <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
        </button>
      </div>

      {/* Scroll Indicator */}
      <div 
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center animate-bounce opacity-50 cursor-pointer" 
        onClick={() => navigate('/dashboard')}
      >
        <span className="text-[10px] font-black uppercase tracking-[0.3em] text-cyan-400 mb-2">Enter Dashboard</span>
        <ArrowDown size={20} className="text-cyan-400" />
      </div>

    </section>
  );
};

export default Hero;
