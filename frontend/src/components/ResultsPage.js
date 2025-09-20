import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, User, MessageCircle, AlertTriangle, Heart } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Footer from './Footer';

const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const data = location.state?.data;

  if (!data) {
    navigate('/');
    return null;
  }

  const { profile, analytics, insights, pdf_path } = data;

  // Add null checks to prevent crashes
  if (!analytics || !analytics.sub_label_distribution) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-4">Analysis Error</h2>
          <p className="text-gray-600 mb-4">Unable to load analysis results. Please try again.</p>
          <button 
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  // Prepare donut chart data for sub-labels
  const subLabelData = Object.entries(analytics.sub_label_distribution).map(([label, count]) => ({
    name: label.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: count,
    percentage: ((count / analytics.total_comments) * 100).toFixed(1)
  }));

  // Prepare bar chart data for main sentiments - ensure all 4 sentiments are shown
  const allSentiments = ['positive', 'negative', 'neutral', 'yellow_flag'];
  const sentimentData = allSentiments.map(sentiment => ({
    sentiment: sentiment === 'yellow_flag' ? 'Yellow Flag' : sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
    count: analytics.sentiment_distribution[sentiment] || 0,
    percentage: (((analytics.sentiment_distribution[sentiment] || 0) / analytics.total_comments) * 100).toFixed(1)
  }));

  // Color schemes for charts
  const sentimentColors = {
    'Positive': '#10B981',
    'Negative': '#EF4444', 
    'Neutral': '#6B7280',
    'Yellow Flag': '#F59E0B'
  };

  const subLabelColors = [
    '#10B981', '#34D399', '#6EE7B7', '#A7F3D0', // Greens for positive
    '#EF4444', '#F87171', '#FCA5A5', '#FECACA', // Reds for negative
    '#6B7280', '#9CA3AF', '#D1D5DB', '#E5E7EB', // Grays for neutral
    '#F59E0B', '#FBBF24', '#FCD34D', '#FDE68A'  // Yellows for flags
  ];

  const handleDownloadPDF = () => {
    if (pdf_path) {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      window.open(`${apiUrl}${pdf_path}`, '_blank');
    }
  };

  return (
    <div className="min-h-screen" style={{background: 'transparent'}}>
      {/* Navigation */}
      <nav className="vite-nav sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <button
              onClick={() => navigate('/')}
              className="flex items-center text-gray-300 hover:text-white font-medium transition-colors"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back to Analysis
            </button>
            <div className="flex items-center">
              <span className="text-xl font-bold text-white">CyberPrint Results</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pt-24">
        {/* Profile Section */}
        <div className="vite-card p-8 mb-8 shadow-2xl">
          <div className="flex items-center justify-center">
            <div className="flex items-center space-x-6">
              <div className="w-20 h-20 vite-gradient rounded-2xl flex items-center justify-center">
                <User className="w-10 h-10 text-white" />
              </div>
              <div className="text-center">
                <h1 className="text-3xl font-bold text-white">{profile.username}</h1>
                <p className="text-gray-300 capitalize text-lg">{profile.platform === 'youtube' ? 'YouTube Channel Analysis' : `${profile.platform} Profile`}</p>
                <div className="flex items-center justify-center mt-3 space-x-6 text-sm text-gray-400">
                  <div className="flex items-center">
                    <MessageCircle className="w-4 h-4 mr-1" />
                    {analytics.total_comments} {profile.platform === 'youtube' ? 'viewer comments analyzed' : 'comments analyzed'}
                  </div>
                  {analytics.mental_health_warnings > 0 && profile.platform !== 'youtube' && (
                    <div className="flex items-center text-red-600">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      {analytics.mental_health_warnings} mental health alerts
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Donut Chart - Sub-labels */}
          <div className="vite-card p-8 shadow-2xl">
            <h2 className="text-xl font-bold text-white mb-6 text-center">Detailed Sentiment Breakdown</h2>
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={subLabelData}
                  cx="50%"
                  cy="45%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {subLabelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={subLabelColors[index % subLabelColors.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value, name) => [`${value} comments (${subLabelData.find(d => d.name === name)?.percentage}%)`, name]}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={80}
                  wrapperStyle={{
                    paddingTop: '20px',
                    fontSize: '12px'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Bar Chart - Main Sentiments */}
          <div className="vite-card p-8 shadow-2xl">
            <h2 className="text-xl font-bold text-white mb-6 text-center">Main Sentiment Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sentiment" />
                <YAxis />
                <Tooltip formatter={(value, name) => [`${value} comments`, 'Count']} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={sentimentColors[entry.sentiment] || '#6B7280'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Insights Section */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* If This Is You */}
          <div className="vite-card p-8 shadow-2xl">
            <div className="flex items-center mb-6">
              <Heart className="w-6 h-6 text-red-400 mr-3" />
              <h2 className="text-xl font-bold text-white">If This Is You</h2>
            </div>
            <div className="space-y-4 mb-6">
              {insights.if_this_is_you.map((tip, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full mt-2 flex-shrink-0"></div>
                  <p className="text-gray-300">{tip}</p>
                </div>
              ))}
            </div>
            {pdf_path && (
              <button
                onClick={handleDownloadPDF}
                className="vite-button w-full py-3 px-6 flex items-center justify-center"
              >
                <Download className="w-5 h-5 mr-2" />
                Download Your CyberPrint Report
              </button>
            )}
          </div>

          {/* If This Is A Stranger */}
          <div className="vite-card p-8 shadow-2xl">
            <div className="flex items-center mb-6">
              <User className="w-6 h-6 text-green-400 mr-3" />
              <h2 className="text-xl font-bold text-white">If This Is Someone Else</h2>
            </div>
            <div className="space-y-4 mb-6">
              {insights.if_this_is_stranger.map((tip, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full mt-2 flex-shrink-0"></div>
                  <p className="text-gray-300">{tip}</p>
                </div>
              ))}
            </div>
            <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4">
              <p className="text-sm text-green-300">
                <strong>💡 Pro Tip:</strong> {profile.platform === 'youtube' ? 'This analysis shows audience feedback patterns that can help content creators understand their community better.' : 'Share CyberPrint with them so they can get their own personalized insights and improve their online communication!'}
              </p>
            </div>
          </div>
        </div>

        {/* Statistics Summary */}
        <div className="mt-8 vite-card p-8 shadow-2xl">
          <h2 className="text-xl font-bold text-white mb-6 text-center">Analysis Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-4xl font-bold text-gradient">{analytics.total_comments}</div>
              <div className="text-sm text-gray-400">Total Comments</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-green-400">
                {((analytics.sentiment_distribution.positive || 0) / analytics.total_comments * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-400">Positive</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-red-400">
                {((analytics.sentiment_distribution.negative || 0) / analytics.total_comments * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-400">Negative</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-yellow-400">{analytics.yellow_flags}</div>
              <div className="text-sm text-gray-400">Yellow Flags</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  );
};

export default ResultsPage;
