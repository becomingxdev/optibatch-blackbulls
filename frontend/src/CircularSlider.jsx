import React, { useState, useRef, useEffect, useCallback } from 'react';

const CircularSlider = ({ value, onChange, min = 0, max = 100 }) => {
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef(null);

  // SVG dimensions (internal viewBox scale)
  const size = 200;
  const radius = 80;
  const cx = size / 2;
  const cy = size / 2;
  
  // Create an arc from -135deg to +135deg (bottom gap is 90 degrees)
  const startAngle = -135;
  const endAngle = 135;
  const totalAngle = endAngle - startAngle;

  const polarToCartesian = (centerX, centerY, radius, angleInDegrees) => {
    const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
    return {
      x: centerX + (radius * Math.cos(angleInRadians)),
      y: centerY + (radius * Math.sin(angleInRadians))
    };
  };

  const getArcPath = (x, y, radius, startAngle, endAngle) => {
    const start = polarToCartesian(x, y, radius, endAngle);
    const end = polarToCartesian(x, y, radius, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
    return [
      "M", start.x, start.y, 
      "A", radius, radius, 0, largeArcFlag, 0, end.x, end.y
    ].join(" ");
  };

  // Convert click/drag coordinates to angle
  const calculateValueFromEvent = useCallback((e) => {
    if (!containerRef.current) return;
    
    // Get EXACT screen center of the SVG element, disregarding CSS padding/scaling
    const rect = containerRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const pointerX = e.clientX || e.touches?.[0]?.clientX;
    const pointerY = e.clientY || e.touches?.[0]?.clientY;

    if (pointerX === undefined || pointerY === undefined) return;

    // Vector from true screen center to pointer
    const x = pointerX - centerX;
    const y = pointerY - centerY;

    // Angle calculation from top-center (0 degrees)
    let angle = Math.atan2(y, x) * 180 / Math.PI + 90;
    
    // Keep angle between -180 and +180
    if (angle > 180) angle -= 360; 

    // Dead-zone calculation: If in the bottom 90 degree gap, snap to closest edge safely
    if (angle > endAngle && angle <= 180) {
        angle = endAngle;
    } else if (angle < startAngle && angle >= -180) {
        angle = startAngle;
    }

    // Convert valid angle distance into a percentage (0 to 1)
    const percentage = (angle - startAngle) / totalAngle;
    
    // Map percentage to actual value
    const newValue = Math.round(min + (percentage * (max - min)));
    
    // Safety clamp
    if (newValue >= min && newValue <= max) {
        onChange(newValue);
    }
  }, [startAngle, endAngle, totalAngle, min, max, onChange]);

  // Handle Dragging State & Global Mouse Events
  useEffect(() => {
    const handleUp = () => setIsDragging(false);
    
    const handleMove = (e) => {
      if (isDragging) {
        // Prevent accidental highlighting/scrolling while adjusting the dial
        if (e.cancelable) e.preventDefault(); 
        calculateValueFromEvent(e);
      }
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMove, { passive: false });
      document.addEventListener('mouseup', handleUp);
      document.addEventListener('touchmove', handleMove, { passive: false });
      document.addEventListener('touchend', handleUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('touchend', handleUp);
    };
  }, [isDragging, calculateValueFromEvent]);

  // Math for rendering the track fill and moving the thumb dot
  const percentage = (value - min) / (max - min);
  const currentAngle = startAngle + (percentage * totalAngle);

  // Dash Arrays for track stylings
  const fullArcLength = (totalAngle / 360) * (2 * Math.PI * radius);
  const filledLength = percentage * fullArcLength;

  return (
    <div 
      className="relative flex flex-col items-center justify-center select-none touch-none"
      ref={containerRef}
      onMouseDown={(e) => { setIsDragging(true); calculateValueFromEvent(e); }}
      onTouchStart={(e) => { setIsDragging(true); calculateValueFromEvent(e); }}
      style={{ touchAction: 'none' }} // Crucial for mobile dial dragging
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="overflow-visible drop-shadow-[0_0_15px_rgba(45,212,191,0.15)] block">
        
        {/* Background Track Arc (Dim) */}
        <path
          d={getArcPath(cx, cy, radius, startAngle, endAngle)}
          fill="none"
          stroke="#1e293b" 
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Inner Dashed Ticks (Decorative) */}
        <path
          d={getArcPath(cx, cy, radius - 15, startAngle, endAngle)}
          fill="none"
          stroke="#334155" 
          strokeWidth="3"
          strokeDasharray="2 6"
        />

        {/* Filled Track Arc (Teal) */}
        <path
          d={getArcPath(cx, cy, radius, startAngle, endAngle)}
          fill="none"
          stroke="#2dd4bf" 
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${fullArcLength}`}
          strokeDashoffset={fullArcLength - filledLength}
          className={!isDragging ? "transition-all duration-100 ease-out" : ""} // Smooth jump when clicked, instant when dragged
        />

        {/* Thumb (Interactive Dot) */}
        <circle
          cx={polarToCartesian(cx, cy, radius, currentAngle).x}
          cy={polarToCartesian(cx, cy, radius, currentAngle).y}
          r="10"
          fill="#2dd4bf"
          stroke="#0f172a"
          strokeWidth="3"
          className={`cursor-pointer transition-transform duration-100 ease-out ${isDragging ? "scale-125" : "hover:scale-125"} ${!isDragging ? "transition-[cx,cy] duration-100" : ""}`}
          style={{ filter: "drop-shadow(0 0 5px rgba(45,212,191,0.8))" }}
        />
      </svg>
      
      {/* Center Value Display */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="text-4xl font-black text-cyan-400 font-mono tracking-tighter drop-shadow-md">{value}%</span>
        <span className="text-[9px] uppercase tracking-[0.2em] font-bold text-slate-500 mt-1">Efficiency</span>
      </div>
    </div>
  );
};

export default CircularSlider;
