import React from 'react';
import { Users, Code, Server, BrainCircuit } from 'lucide-react';

const TeamMember = ({ name, role, icon: Icon, delay }) => (
  <div 
    className="glass-panel p-6 rounded-3xl flex flex-col items-center text-center transition-all duration-500 hover:-translate-y-2 hover:shadow-[0_0_30px_rgba(45,212,191,0.2)] group"
    style={{ animationDelay: `${delay}ms` }}
  >
    <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50 flex items-center justify-center mb-6 shadow-inner group-hover:border-cyan-500/50 transition-colors">
      <Icon size={32} className="text-cyan-400 group-hover:text-cyan-300 group-hover:scale-110 transition-transform" />
    </div>
    <h4 className="text-xl font-black text-white tracking-tight mb-2">{name}</h4>
    <div className="bg-cyan-500/10 border border-cyan-500/20 px-3 py-1 rounded-full">
      <p className="text-xs font-bold text-cyan-400 uppercase tracking-widest">{role}</p>
    </div>
  </div>
);

const AboutUs = () => {
  return (
    <section className="min-h-screen w-full flex flex-col items-center justify-center py-24 px-6 relative z-10">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#050505] to-[#0a0f1c] -z-10" />
      <div className="max-w-5xl w-full mx-auto">
        <div className="text-center mb-20">
          <div className="inline-flex items-center gap-3 bg-slate-800/50 border border-slate-700/50 px-4 py-2 rounded-full mb-6">
            <Users size={16} className="text-cyan-400" />
            <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">Meet The Developers</span>
          </div>
          <h2 className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-slate-400 tracking-tighter mb-6">
            Team BlackBull
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
            Pioneering the next generation of intelligent industrial optimization.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <TeamMember name="Satyam" role="Backend Infrastructure" icon={Server} delay={0} />
          <TeamMember name="Kunal" role="Frontend Architecture" icon={Code} delay={100} />
          <TeamMember name="Dev Desai" role="AI / ML Models" icon={BrainCircuit} delay={200} />
        </div>
        <div className="mt-32 flex justify-center w-full opacity-30">
          <div className="h-px w-full max-w-sm bg-gradient-to-r from-transparent via-cyan-500 to-transparent"></div>
        </div>
      </div>
    </section>
  );
};

export default AboutUs;
