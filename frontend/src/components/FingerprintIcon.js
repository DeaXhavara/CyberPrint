import React, { memo } from 'react';

const FingerprintIcon = memo(({ className = "", size = 120 }) => {
  return (
    <div className={`fingerprint-container ${className}`} style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 120 120"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="fingerprint-svg"
      >
        
        {/* First U-shaped arc */}
        <path
          className="fingerprint-line ridge-2"
          d="M25 80 L25 50 C25 33, 40 20, 60 20 C80 20, 95 33, 95 50 L95 87"
          stroke="url(#gradient2)"
          strokeWidth="0.5"
          strokeLinecap="round"
          fill="none"
        />

        {/* Second U-shaped arc */}
        <path
          className="fingerprint-line ridge-3"
          d="M35 90 L35 50 C35 40, 46 30, 60 30 C74 30, 85 40, 85 50 L85 80"
          stroke="url(#gradient3)"
          strokeWidth="0.9"
          strokeLinecap="round"
          fill="none"
        />

        {/* Third U-shaped arc */}
        <path
          className="fingerprint-line ridge-4"
          d="M45 100 L45 50 C45 46, 52 40, 60 40 C68 40, 75 46, 75 50 L75 90 C75 95, 85 105, 95 93"
          stroke="url(#gradient4)"
          strokeWidth="0.5"
          strokeLinecap="round"
          fill="none"
        />

        {/* Innermost short arc */}
        <path
          className="fingerprint-line ridge-5"
          d="M55 110 L55 50 C55 48, 57 46, 60 46 C63 46, 65 48, 65 50 L65 95"
          stroke="url(#gradient5)"
          strokeWidth="1.3"
          strokeLinecap="round"
          fill="none"
        />

        {/* Center vertical line */}
        <path
          className="fingerprint-line center-line"
          d="M60 46 L60 95"
          stroke="url(#gradient6)"
          strokeWidth="1.9"
          strokeLinecap="round"
          fill="none"
        />

        {/* Center dot */}
        <circle
          cx="65"
          cy="105"
          r="1.5"
          fill="url(#centerGradient)"
          className="fingerprint-center"
        />
        
        {/* Gradient definitions */}
        <defs>
          <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.9" />
            <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.8" />
          </linearGradient>
          
          <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8" />
          </linearGradient>
          
          <linearGradient id="gradient3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.7" />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.7" />
          </linearGradient>
          
          <linearGradient id="gradient4" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
            <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.6" />
          </linearGradient>
          
          <linearGradient id="gradient5" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.5" />
            <stop offset="50%" stopColor="#3b82f6" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.5" />
          </linearGradient>
          
          <linearGradient id="gradient6" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.8" />
          </linearGradient>
          
          <radialGradient id="centerGradient">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
            <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.8" />
          </radialGradient>
        </defs>
      </svg>
    </div>
  );
});

export default FingerprintIcon;
