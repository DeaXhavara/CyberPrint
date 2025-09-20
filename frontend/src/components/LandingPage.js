import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Shield, Search, TrendingUp } from 'lucide-react';
import FingerprintIcon from './FingerprintIcon';
import Footer from './Footer';
import axios from 'axios';
import gsap from 'gsap';

const LandingPage = () => {
  const [url, setUrl] = useState('');
  // const [platform, setPlatform] = useState('reddit');
  const [numComments, setNumComments] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [scrolled, setScrolled] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 50;
      setScrolled(isScrolled);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/analyze`, {
        url: url,
        num_comments: numComments
      });

      // Navigate to results page with data
      navigate('/results', { state: { data: response.data } });
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        setError(detail.map(d => d.msg || JSON.stringify(d)).join(' | '));
      } else if (detail && typeof detail === 'object') {
        setError(detail.msg || JSON.stringify(detail));
      } else {
        setError(detail || 'Analysis failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  }, [url, numComments, navigate]);

  // Retro Futuristic Button component based on CodePen design
  const RetroButton = ({ onClick, disabled, loading }) => {
    const btnRef = useRef(null);
    const rectsRef = useRef([]);

    useEffect(() => {
      const rects = rectsRef.current;
      if (!rects.length) return;

      // Initial state - all rects hidden to the left
      gsap.set(rects, { x: "-100%" });

      const handleMouseEnter = () => {
        if (disabled) return;
        gsap.to(gsap.utils.shuffle(rects), {
          duration: 0.8,
          ease: "elastic.out(1, 0.3)",
          x: "0%",
          stagger: 0.02
        });
      };

      const handleMouseLeave = () => {
        if (disabled) return;
        gsap.to(gsap.utils.shuffle(rects), {
          duration: 0.6,
          ease: "power2.out",
          x: "-100%",
          stagger: 0.01
        });
      };

      const btn = btnRef.current;
      btn.addEventListener('mouseenter', handleMouseEnter);
      btn.addEventListener('mouseleave', handleMouseLeave);

      return () => {
        btn.removeEventListener('mouseenter', handleMouseEnter);
        btn.removeEventListener('mouseleave', handleMouseLeave);
      };
    }, [disabled]);

    return (
      <>
        <style>{`
          .retro-button {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 1rem 2rem;
            font-size: 1rem;
            font-weight: 500;
            font-family: 'IBM Plex Mono', monospace;
            font-style: italic;
            color: #fff;
            background: #3d3e4a;
            border: 2px solid #555;
            border-radius: 8px;
            cursor: pointer;
            overflow: hidden;
            user-select: none;
            width: 100%;
            text-align: center;
            transition: all 0.3s ease;
            box-sizing: border-box;
            text-transform: uppercase;
            letter-spacing: 1px;
          }
          .retro-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
          }
          .retro-button:hover:not(:disabled) {
            background: #4a4b58;
            border-color: #666;
          }
          .retro-button svg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
          }
          .retro-button rect {
            fill: #6c63ff;
            opacity: 0.8;
          }
          .retro-button .content {
            position: relative;
            z-index: 10;
            display: flex;
            align-items: center;
            justify-content: center;
          }
          .retro-button .spinner {
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid #fff;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
          }
          @keyframes spin {
            0% { transform: rotate(0deg);}
            100% { transform: rotate(360deg);}
          }
        `}</style>
        <button
          ref={btnRef}
          className="retro-button"
          onClick={onClick}
          disabled={disabled}
          type="submit"
        >
          <svg height="100%" width="100%" xmlns="http://www.w3.org/2000/svg">
            {Array.from({ length: 22 }, (_, i) => (
              <rect
                key={i}
                ref={el => rectsRef.current[i] = el}
                x="-100%"
                y={i * 2}
                width="100%"
                height="2"
              />
            ))}
          </svg>
          <div className="content">
            {loading && <div className="spinner" aria-hidden="true"></div>}
            {loading ? 'Analyzing...' : 'Analyze Profile'}
          </div>
        </button>
      </>
    );
  };

  return (
    <div className="min-h-screen w-full overflow-x-hidden" style={{ background: 'transparent' }}>
      {/* Navigation */}
      <nav className={`vite-nav ${scrolled ? 'scrolled' : ''}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <img 
                src="/cyberPrint.png" 
                alt="CyberPrint Logo" 
                className="h-28 max-w-full w-auto"
              />
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <button 
                onClick={() => navigate('/about')}
                className="text-gray-300 hover:text-white transition-colors"
              >
                About
              </button>
              <button 
                onClick={() => navigate('/contact')}
                className="text-gray-300 hover:text-white transition-colors"
              >
                Contact
              </button>
              <a 
                href="https://github.com/DeaXhavara" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition-colors"
              >
                <img 
                  src="/github-white.svg" 
                  alt="GitHub" 
                  className="w-6 h-6"
                />
              </a>
              <a 
                href="https://www.linkedin.com/in/deaxhavara/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition-colors"
              >
                <img 
                  src="/linkedin-white.svg" 
                  alt="LinkedIn" 
                  className="w-6 h-6"
                />
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pt-32">
        <div className="text-center mb-16">
          {/* Hero Image */}
          <div className="mb-12 flex justify-center hero-glow">
            <div className="relative">
              <FingerprintIcon size={400} className="fingerprint-hero transform hover:scale-105 transition-transform duration-300" />
            </div>
          </div>

          {/* Hero Text */}
          <h1 className="text-6xl font-black text-white mb-6 leading-tight break-words" style={{textDecoration: 'none', borderBottom: 'none', outline: 'none'}}>
            Your online behavior, <span className="text-gradient" style={{textDecoration: 'none', borderBottom: 'none', outline: 'none', display: 'inline-block'}}>decoded</span>
          </h1>
          <p className="text-xl text-gray-300 mb-16 max-w-4xl mx-auto leading-relaxed">
            Discover insights about your digital footprint with <span className="text-gradient font-semibold">AI Agent for sentiment analysis</span>. 
            Understand how you communicate online and get personalized tips for improvement.
          </p>
          
          {/* Feature highlights */}
          <div className="flex justify-center items-center space-x-8 mb-12 text-sm text-gray-400">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              <span>AI-Powered Analysis</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
              <span>Privacy Focused</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
              <span>Instant Results</span>
            </div>
          </div>

          {/* Analysis Form */}
          <div className="max-w-2xl mx-auto">
            <div className="glass-card">
              <div className="glass-card-inner p-8">
              <form onSubmit={handleSubmit} className="space-y-6">
                {error && (
                  <div className="bg-red-900/20 border border-red-500/30 text-red-300 px-4 py-3 rounded-lg">
                    {error}
                  </div>
                )}
                
                <div>
                  <label htmlFor="url" className="block text-sm font-medium text-gray-300 mb-2">
                    Enter profile/channel URL
                  </label>
                  <div className="relative group">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5 group-focus-within:text-blue-400 transition-colors duration-200" />
                    <input
                      type="url"
                      id="url"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      placeholder="https://reddit.com/u/username or https://youtube.com/@channel"
                      className="vite-input w-full pl-10 pr-4 py-3"
                      required
                    />
                    <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-blue-500/10 to-purple-500/10 opacity-0 group-focus-within:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                  </div>
                </div>


                <div>
                  <label htmlFor="comments" className="block text-sm font-medium text-gray-300 mb-2">
                    Number of comments to analyze
                  </label>
                  <select
                    id="comments"
                    value={numComments}
                    onChange={(e) => setNumComments(parseInt(e.target.value))}
                    className="vite-input w-full px-4 py-3"
                  >
                    <option value={25}>25 comments</option>
                    <option value={50}>50 comments</option>
                    <option value={100}>100 comments</option>
                    <option value={200}>200 comments</option>
                  </select>
                </div>

                <RetroButton 
                  onClick={handleSubmit} 
                  disabled={loading} 
                  loading={loading} 
                />
              </form>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-8 mt-24">
          <div className="glass-card">
            <div className="glass-card-inner">
              <div className="glass-icon-wrapper">
                <Brain className="w-10 h-10 text-white" />
              </div>
              <h3 className="glass-card-title">AI-Powered Analysis</h3>
              <p className="glass-card-text">Sentiment analysis using state-of-the-art <span className="text-gradient">DeBERTa</span> machine learning models.</p>
            </div>
          </div>
          
          <div className="glass-card">
            <div className="glass-card-inner">
              <div className="glass-icon-wrapper">
                <Shield className="w-10 h-10 text-white" />
              </div>
              <h3 className="glass-card-title">Privacy First</h3>
              <p className="glass-card-text">Your data is processed securely and never stored permanently. <span className="text-gradient">Zero tracking</span>.</p>
            </div>
          </div>
          
          <div className="glass-card">
            <div className="glass-card-inner">
              <div className="glass-icon-wrapper">
                <TrendingUp className="w-10 h-10 text-white" />
              </div>
              <h3 className="glass-card-title">Actionable Insights</h3>
              <p className="glass-card-text">Get personalized recommendations to improve your online presence with <span className="text-gradient">real-time feedback</span>.</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  );
};

export default LandingPage;
