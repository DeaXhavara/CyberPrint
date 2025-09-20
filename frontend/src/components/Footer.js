import React from 'react';
import { Heart, Github, Linkedin, Mail, Shield, Code, Database } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="mt-16 border-t border-gray-800/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="md:col-span-2">
            <div className="flex items-center mb-4">
              <img 
                src="/cyberPrint.png" 
                alt="CyberPrint Logo" 
                className="h-16 w-auto mr-3"
              />
            </div>
            <p className="text-gray-400 mb-4 max-w-md">
              Advanced AI-powered sentiment analysis for your digital footprint. 
              Understand your online behavior and improve your digital wellbeing.
            </p>
            <div className="flex items-center space-x-4">
              <a 
                href="https://github.com/DeaXhavara" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5" />
              </a>
              <a 
                href="https://www.linkedin.com/in/deaxhavara/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="LinkedIn"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a 
                href="mailto:deaxhavara@gmail.com" 
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="Email"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Features */}
          <div>
            <h3 className="text-white font-semibold mb-4">Features</h3>
            <ul className="space-y-2 text-gray-400">
              <li className="flex items-center">
                <Code className="w-4 h-4 mr-2 text-blue-400" />
                <span className="text-sm">AI Sentiment Analysis</span>
              </li>
              <li className="flex items-center">
                <Database className="w-4 h-4 mr-2 text-purple-400" />
                <span className="text-sm">DeBERTa ML Models</span>
              </li>
              <li className="flex items-center">
                <Shield className="w-4 h-4 mr-2 text-green-400" />
                <span className="text-sm">Privacy Focused</span>
              </li>
            </ul>
          </div>

          {/* Legal & Info */}
          <div>
            <h3 className="text-white font-semibold mb-4">Information</h3>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li>
                <a href="/about" className="hover:text-white transition-colors">
                  About CyberPrint
                </a>
              </li>
              <li>
                <a href="/contact" className="hover:text-white transition-colors">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-8 border-t border-gray-800/50">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-gray-400 text-sm mb-4 md:mb-0">
              © {new Date().getFullYear()} CyberPrint. Made with{' '}
              <Heart className="w-4 h-4 inline text-red-400" />{' '}
              for digital wellbeing.
            </div>
            <div className="text-gray-500 text-xs">
              Powered by advanced AI • Privacy-first design • Open source
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
