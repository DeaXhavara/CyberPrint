import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ResultsPage from './components/ResultsPage';
import AboutPage from './components/AboutPage';
import ContactPage from './components/ContactPage';
import AnimatedBackground from './components/AnimatedBackground';
import './index.css';

function App() {
  return (
    <Router>
      <div className="App">
        <AnimatedBackground />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
