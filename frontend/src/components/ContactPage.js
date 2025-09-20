import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Mail, Phone, MapPin, Shield, Zap, Search, TrendingUp, Eye, BarChart3, Lock, Database, Code, Users } from 'lucide-react';
import Footer from './Footer';

const ContactPage = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission here
    console.log('Form submitted:', formData);
    // Create mailto link with form data
    const subject = encodeURIComponent(formData.subject || 'Contact from CyberPrint');
    const body = encodeURIComponent(
      `Name: ${formData.name}\n` +
      `Email: ${formData.email}\n\n` +
      `Message:\n${formData.message}`
    );
    const mailtoLink = `mailto:deaxhavara@gmail.com?subject=${subject}&body=${body}`;
    
    // Open email client
    window.location.href = mailtoLink;
    
    // Reset form
    setFormData({ name: '', email: '', subject: '', message: '' });
    alert('Opening your email client to send the message...');
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

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
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/30 mb-8">
            <Mail className="w-5 h-5 text-green-400 mr-2" />
            <span className="text-green-300 font-medium">Connect With Us</span>
          </div>
          <h1 className="text-6xl font-black text-white mb-8 leading-tight">
            Get in <span className="text-gradient">Touch</span>
          </h1>
          <p className="text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed font-light">
            Have questions, feedback, or need support? We'd love to hear from you. 
            Reach out and let's improve digital communication together.
          </p>
        </header>

        {/* Contact Information */}
        <section className="mb-16">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">Get in Touch</h2>
              <div className="w-32 h-px bg-gradient-to-r from-blue-400 to-transparent"></div>
            </div>
            <div className="flex-1 space-y-6 text-lg text-gray-300 leading-relaxed">
              <p>
                For <span className="text-green-400 font-semibold">questions, feedback, or collaboration opportunities</span>, 
                feel free to reach out via email at 
                <a href="mailto:deaxhavara@gmail.com" className="text-green-400 hover:text-green-300 underline ml-1">deaxhavara@gmail.com</a>. 
                I'd love to hear from you and discuss CyberPrint or potential projects.
              </p>
              <p>
                You can also connect with me on <span className="text-blue-400 font-semibold">LinkedIn</span> for professional networking 
                or check out my other projects on <span className="text-purple-400 font-semibold">GitHub</span>. 
                I'm always open to discussing AI, machine learning, and innovative tech solutions.
              </p>
            </div>
          </div>
        </section>

        {/* Contact Form */}
        <section className="mb-16">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">Send us a Message</h2>
              <div className="w-40 h-px bg-gradient-to-r from-green-400 to-transparent"></div>
            </div>
          </div>
          
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto space-y-6 text-left">
            <fieldset className="grid md:grid-cols-2 gap-6 border-0">
              <div>
                <label htmlFor="name" className="block text-white font-medium mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  placeholder="Dea Xhavara"
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-white font-medium mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  placeholder="deaxhavara@gmail.com"
                />
              </div>
            </fieldset>
            
            <div>
              <label htmlFor="subject" className="block text-white font-medium mb-2">
                Subject
              </label>
              <input
                type="text"
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                placeholder="What's this about?"
              />
            </div>
            
            <div>
              <label htmlFor="message" className="block text-white font-medium mb-2">
                Message
              </label>
              <textarea
                id="message"
                name="message"
                value={formData.message}
                onChange={handleChange}
                required
                rows={6}
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
                placeholder="Tell us more about your inquiry..."
              />
            </div>
            
            <div className="text-center pt-4">
              <button
                type="submit"
                className="vite-button py-4 px-12 text-xl font-bold"
              >
                Send Message
              </button>
            </div>
          </form>
        </section>

        {/* Social Links */}
        <section className="mb-16">
          <div className="flex items-start max-w-6xl mx-auto mb-12">
            <div className="flex-shrink-0 mr-12 min-w-fit">
              <h2 className="text-4xl font-bold text-white mb-4 whitespace-nowrap">Connect With Me</h2>
              <div className="w-36 h-px bg-gradient-to-r from-purple-400 to-transparent"></div>
            </div>
            <div className="flex-1">
              <p className="text-lg text-gray-300 leading-relaxed">
                Let's connect and explore opportunities together
              </p>
            </div>
          </div>
          
          <nav className="flex justify-center space-x-12 max-w-4xl mx-auto" aria-label="Social media links">
            <a
              href="https://github.com/DeaXhavara"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center p-6 hover:scale-110 transition-transform"
              aria-label="GitHub"
            >
              <div className="glass-icon-wrapper w-16 h-16 mb-4">
                <img src="/github-white.svg" alt="GitHub" className="w-8 h-8" />
              </div>
              <span className="text-white font-semibold text-lg">GitHub</span>
              <span className="text-gray-400 text-sm mt-1">View Projects</span>
            </a>
            
            <a
              href="https://www.linkedin.com/in/deaxhavara/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col items-center p-6 hover:scale-110 transition-transform"
              aria-label="LinkedIn"
            >
              <div className="glass-icon-wrapper w-16 h-16 mb-4">
                <img src="/linkedin-white.svg" alt="LinkedIn" className="w-8 h-8" />
              </div>
              <span className="text-white font-semibold text-lg">LinkedIn</span>
              <span className="text-gray-400 text-sm mt-1">Professional Network</span>
            </a>
            
            <a
              href="mailto:deaxhavara@gmail.com"
              className="flex flex-col items-center p-6 hover:scale-110 transition-transform"
            >
              <div className="glass-icon-wrapper w-16 h-16 mb-4">
                <Mail className="w-8 h-8 text-green-400" />
              </div>
              <span className="text-white font-semibold text-lg">Email</span>
              <span className="text-gray-400 text-sm mt-1">Direct Contact</span>
            </a>
          </nav>
        </section>


        {/* Call to Action */}
        <section className="text-center mb-20">
          <div className="w-full h-px bg-gradient-to-r from-transparent via-gray-600 to-transparent mb-16"></div>
          <h2 className="text-5xl font-black text-white mb-8">
            Ready to Transform Your <span className="text-gradient">Digital Communication?</span>
          </h2>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
            Experience the future of AI-powered sentiment analysis and digital communication insights. 
            Try CyberPrint today and discover what your digital footprint reveals.
          </p>
          <button
            onClick={() => navigate('/')}
            className="vite-button py-4 px-12 text-xl font-bold"
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

export default ContactPage;
