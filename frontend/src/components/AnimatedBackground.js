import React, { useEffect, useRef, memo } from 'react';

const AnimatedBackground = memo(() => {
  const containerRef = useRef(null);

  useEffect(() => {
    // Generate random star positions with inline styles for guaranteed visibility
    const generateStars = (count) => {
      const shadows = [];
      for (let i = 0; i < count; i++) {
        const x = Math.floor(Math.random() * window.innerWidth);
        const y = Math.floor(Math.random() * (window.innerHeight + 2000)); // Extended range for animation
        shadows.push(`${x}px ${y}px 0px #FFFFFF`);
      }
      return shadows.join(', ');
    };

    const applyStars = () => {
      if (containerRef.current) {
        const starsElement = containerRef.current.querySelector('.stars');
        const stars2Element = containerRef.current.querySelector('.stars2');
        const stars3Element = containerRef.current.querySelector('.stars3');

        if (starsElement) {
          const smallStars = generateStars(800);
          starsElement.style.boxShadow = smallStars;
          starsElement.style.background = '#FFFFFF';
        }

        if (stars2Element) {
          const mediumStars = generateStars(400);
          stars2Element.style.boxShadow = mediumStars;
          stars2Element.style.background = '#FFFFFF';
        }

        if (stars3Element) {
          const largeStars = generateStars(200);
          stars3Element.style.boxShadow = largeStars;
          stars3Element.style.background = '#FFFFFF';
        }
      }
    };

    // Initial application
    const timer = setTimeout(applyStars, 200);

    // Regenerate stars less frequently to reduce performance impact
    const interval = setInterval(applyStars, 120000); // Every 2 minutes

    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, []);

  return (
    <div 
      ref={containerRef} 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%)',
        overflow: 'hidden',
        zIndex: -10,
        pointerEvents: 'none'
      }}
    >
      {/* Gradient mask overlay to fade stars from fingerprint area */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          background: 'linear-gradient(to bottom, rgba(27, 39, 53, 1) 0%, rgba(27, 39, 53, 0.98) 15%, rgba(27, 39, 53, 0.95) 25%, rgba(27, 39, 53, 0.9) 35%, rgba(27, 39, 53, 0.8) 45%, rgba(27, 39, 53, 0.65) 55%, rgba(27, 39, 53, 0.45) 65%, rgba(27, 39, 53, 0.25) 75%, rgba(27, 39, 53, 0.1) 85%, rgba(27, 39, 53, 0.02) 92%, rgba(27, 39, 53, 0) 100%)',
          zIndex: 2,
          pointerEvents: 'none'
        }}
      />
      
      <div 
        className="stars"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '1px',
          height: '1px',
          background: '#FFFFFF',
          animation: 'animStar 50s linear infinite',
          zIndex: 1
        }}
      ></div>
      <div 
        className="stars2"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '2px',
          height: '2px',
          background: '#FFFFFF',
          animation: 'animStar 100s linear infinite',
          zIndex: 1
        }}
      ></div>
      <div 
        className="stars3"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '3px',
          height: '3px',
          background: '#FFFFFF',
          animation: 'animStar 150s linear infinite',
          zIndex: 1
        }}
      ></div>
    </div>
  );
});

export default AnimatedBackground;
