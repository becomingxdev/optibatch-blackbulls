import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import SweepExplorer from './pages/SweepExplorer';

const App = () => {
  return (
    <Routes>
      <Route path="/"          element={<Home />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/analytics" element={<Analytics />} />
      <Route path="/sweep-explorer" element={<SweepExplorer />} />
    </Routes>
  );
};

export default App;