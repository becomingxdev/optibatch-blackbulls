import React from 'react';
import ParticlesBackground from '../components/ParticlesBackground';
import Hero from '../components/Hero';
import AboutUs from '../components/AboutUs';

const Home = () => {
  return (
    <div className="bg-[#050505] min-h-screen text-slate-200 font-sans selection:bg-cyan-500/30 relative">
      <ParticlesBackground />
      <Hero />
      <AboutUs />
    </div>
  );
};

export default Home;
