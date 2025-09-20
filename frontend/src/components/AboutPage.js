import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Brain, Shield, Zap, TrendingUp, Database, Bot, Lightbulb, Award, Target, Globe } from 'lucide-react';
import Footer from './Footer';

const AboutPage = () => {
  const navigate = useNavigate();

  return (
    <main className="min-h-screen" style={{background: 'transparent'}}>
      {/* Navigation */}
      <nav className="vite-nav sticky top-0 z-50">
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <header className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate('/')}
              className="flex items-center text-gray-300 hover:text-white font-medium transition-colors"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back to Home
            </button>
            <figure className="flex items-center">
              <img 
                src="/cyberPrint.png" 
                alt="CyberPrint Logo" 
                className="h-28 w-auto"
              />
            </figure>
          </header>
        </section>
      </nav>

      <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 pt-24">
        {/* Hero Section */}
        <header className="text-center mb-20">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30 mb-8">
            <Bot className="w-5 h-5 text-blue-400 mr-2" />
            <span className="text-blue-300 font-medium">Agentic AI for Digital Wellbeing</span>
          </div>
          <h1 className="text-6xl font-black text-white mb-8 leading-tight">
            About <span className="text-gradient">CyberPrint</span>
          </h1>
          <p className="text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed font-light">
            An autonomous AI agent that analyzes, adapts, and evolves to provide meaningful insights into your digital communication patterns.
          </p>
        </header>


        {/* Why CyberPrint? */}
        <section className="mb-20">
          <div className="flex items-start max-w-6xl mx-auto">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">Why CyberPrint?</h2>
              <div className="w-32 h-px bg-gradient-to-r from-blue-400 to-transparent"></div>
            </div>
            <div className="flex-1 space-y-8 text-lg text-gray-300 leading-relaxed">
              <p>
                CyberPrint represents the next evolution in digital communication analysis through <span className="text-blue-400 font-semibold">Agentic AI</span>. 
                Our autonomous system doesn't just analyze data - it actively learns, adapts, and provides intelligent recommendations 
                for your digital wellbeing. Unlike traditional sentiment analysis tools, CyberPrint operates as an independent AI agent 
                that continuously evolves to understand the nuances of human communication.
              </p>
              <p>
                In today's digital landscape, where <span className="text-green-400 font-semibold">mental health awareness</span> has become 
                critical, CyberPrint serves as your intelligent companion. Our system identifies early warning signs in communication 
                patterns, provides personalized mental health resources, and offers actionable insights for professional reputation 
                management. This proactive approach transforms reactive mental health support into preventive digital wellness.
              </p>
              <p>
                Every analysis makes our system smarter through <span className="text-purple-400 font-semibold">continuous evolution</span>. 
                What starts as basic sentiment analysis becomes personalized digital coaching, creating a self-improving platform 
                that grows with digital communication trends and adapts to each user's unique communication style.
              </p>
            </div>
          </div>
        </section>

        {/* Problem & Solution */}
        <section className="mb-20">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">The Problem & Our Solution</h2>
              <div className="w-48 h-px bg-gradient-to-r from-red-400 to-transparent"></div>
            </div>
            <div className="flex-1">
              <p className="text-lg text-gray-300 leading-relaxed">
                Addressing critical challenges in digital communication and mental health awareness.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto relative">
            {/* Vertical separator line */}
            <div className="hidden md:block absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-gray-600 to-transparent transform -translate-x-1/2"></div>
            
            <div className="pr-8">
              <h3 className="text-2xl font-bold text-red-400 mb-6 flex items-center">
                <span className="w-3 h-3 bg-red-400 rounded-full mr-3"></span>
                The Challenge
              </h3>
              <ul className="space-y-4 text-gray-300">
                <li className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">•</span>
                  <span>70% of online harassment goes unrecognized by traditional detection systems</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">•</span>
                  <span>Mental health crises often manifest in digital communication patterns before real-world symptoms</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">•</span>
                  <span>Employers increasingly screen social media, but individuals lack tools to understand their digital footprint</span>
                </li>
                <li className="flex items-start">
                  <span className="text-red-400 mr-3 mt-1">•</span>
                  <span>Current sentiment analysis tools lack context awareness and emotional intelligence</span>
                </li>
              </ul>
            </div>
            
            <div className="pl-8">
              <h3 className="text-2xl font-bold text-green-400 mb-6 flex items-center">
                <span className="w-3 h-3 bg-green-400 rounded-full mr-3"></span>
                Our Solution
              </h3>
              <ul className="space-y-4 text-gray-300">
                <li className="flex items-start">
                  <span className="text-green-400 mr-3 mt-1">•</span>
                  <span><strong>Autonomous AI Agent</strong> that processes communication patterns with human-level understanding</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-3 mt-1">•</span>
                  <span><strong>Early Detection System</strong> for mental health indicators and digital wellbeing concerns</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-3 mt-1">•</span>
                  <span><strong>Professional Footprint Optimization</strong> with actionable insights and recommendations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-green-400 mr-3 mt-1">•</span>
                  <span><strong>Context-Aware Analysis</strong> that understands nuance, sarcasm, and cultural communication patterns</span>
                </li>
              </ul>
            </div>
          </div>
        </section>


        {/* Technical Architecture */}
        <section className="mb-20">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">Technical Architecture</h2>
              <div className="w-40 h-px bg-gradient-to-r from-purple-400 to-transparent"></div>
            </div>
            <div className="flex-1">
              <p className="text-lg text-gray-300 leading-relaxed">
                Built with cutting-edge AI technologies and designed for scalability, privacy, and performance.
              </p>
            </div>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <Brain className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">DeBERTa Agent</h3>
                <p className="glass-card-text text-sm">
                  93.4% accuracy with self-adapting gratitude detection and continuous learning capabilities.
                </p>
              </div>
            </div>
            
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <Database className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Data Pipeline</h3>
                <p className="glass-card-text text-sm">
                  Intelligent multi-platform fetching with privacy preservation and real-time processing.
                </p>
              </div>
            </div>
            
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <Lightbulb className="w-6 h-6 text-yellow-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Insight Engine</h3>
                <p className="glass-card-text text-sm">
                  Autonomous recommendation generation with mental health awareness and professional guidance.
                </p>
              </div>
            </div>
            
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <TrendingUp className="w-6 h-6 text-purple-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Active Learning</h3>
                <p className="glass-card-text text-sm">
                  Continuous improvement through misclassification detection and human-in-the-loop feedback.
                </p>
              </div>
            </div>
            
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <Shield className="w-6 h-6 text-red-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Privacy First</h3>
                <p className="glass-card-text text-sm">
                  Local processing, encrypted data handling, and user-controlled privacy settings.
                </p>
              </div>
            </div>
            
            <div className="glass-card">
              <div className="glass-card-inner p-6 text-center">
                <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                  <Award className="w-6 h-6 text-orange-400" />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">Report Generation</h3>
                <p className="glass-card-text text-sm">
                  Beautiful PDF reports with pastel color schemes and professional typography.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Autonomous AI Agent */}
        <section className="mb-20">
          <div className="glass-card">
            <div className="glass-card-inner p-12">
              <div className="text-center mb-12">
                <div className="glass-icon-wrapper w-16 h-16 mx-auto mb-6">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-4xl font-bold text-white mb-4">Autonomous AI Agent Innovation</h2>
                <p className="glass-card-text text-lg opacity-90 max-w-3xl mx-auto">
                  CyberPrint represents the cutting edge of <strong>HackOmatic 2025: Agentic AI</strong> - 
                  autonomous systems that act, adapt, and evolve to create meaningful impact.
                </p>
              </div>
              
              <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                <div className="text-center">
                  <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                    <Zap className="w-6 h-6 text-yellow-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">Autonomy</h3>
                  <p className="glass-card-text text-sm">
                    Independent processing, decision-making, and insight generation without human intervention. Our AI agents work 24/7 to protect your digital wellbeing.
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                    <Target className="w-6 h-6 text-blue-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">Adaptability</h3>
                  <p className="glass-card-text text-sm">
                    Our system evolves with each analysis, learning from patterns and improving accuracy. What starts as analysis becomes personalized digital coaching.
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="glass-icon-wrapper w-12 h-12 mx-auto mb-4">
                    <Globe className="w-6 h-6 text-green-400" />
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">Impact</h3>
                  <p className="glass-card-text text-sm">
                    Real-world applications in mental health awareness, digital literacy, online safety, and professional development. AI that makes a difference.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="mb-20">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">How It Works</h2>
              <div className="w-32 h-px bg-gradient-to-r from-green-400 to-transparent"></div>
            </div>
            <div className="flex-1 space-y-8 text-lg text-gray-300 leading-relaxed">
              <p>
                <span className="text-blue-400 font-semibold">Step 1: Data Collection</span> - We securely fetch your recent public comments 
                from the platform you specify using official APIs. Your data is processed locally and never stored permanently, 
                ensuring complete privacy and security throughout the analysis process.
              </p>
              <p>
                <span className="text-green-400 font-semibold">Step 2: AI Analysis</span> - Our DeBERTa model, achieving 93.4% accuracy, 
                analyzes sentiment, tone, and communication patterns using state-of-the-art natural language processing techniques. 
                The system identifies emotional indicators, communication styles, and potential areas of concern.
              </p>
              <p>
                <span className="text-purple-400 font-semibold">Step 3: Insights Generation</span> - The autonomous AI agent generates 
                personalized recommendations, mental health resources, and actionable tips for improvement. This includes professional 
                reputation optimization, digital wellness suggestions, and early intervention recommendations when needed.
              </p>
              <p>
                <span className="text-yellow-400 font-semibold">Step 4: Beautiful Reports</span> - Receive comprehensive interactive 
                visualizations and downloadable PDF reports with your complete analysis. The reports include sentiment breakdowns, 
                communication pattern insights, and personalized action plans for digital wellbeing improvement.
              </p>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="text-center">
          <button
            onClick={() => navigate('/')}
            className="vite-button py-4 px-8 text-lg font-bold"
          >
            Try CyberPrint Now
          </button>
        </section>
      </section>
      
      {/* Footer */}
      <Footer />
    </main>
  );
};

export default AboutPage;
