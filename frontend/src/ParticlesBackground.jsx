import React, { useEffect, useRef } from 'react';

/**
 * WaveParticles — canvas particle field with cursor ripple effect.
 * Particles drift slowly, and when the cursor moves they are pushed
 * outward in an expanding wave ring from the mouse position.
 */
const ParticlesBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let rafId;

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    let mouse = { x: -9999, y: -9999, moved: false };
    const onMove = (e) => {
      const x = e.clientX ?? e.touches?.[0]?.clientX;
      const y = e.clientY ?? e.touches?.[0]?.clientY;
      if (x == null) return;
      mouse.x = x; mouse.y = y; mouse.moved = true;
    };
    const onLeave = () => { mouse.x = -9999; mouse.y = -9999; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove',  onMove, { passive: true });
    window.addEventListener('mouseleave', onLeave);

    const ripples = [];
    let lastRippleTime = 0;

    const spawnRipple = () => {
      if (mouse.x === -9999) return;
      const now = performance.now();
      if (now - lastRippleTime < 60) return;
      lastRippleTime = now;
      ripples.push({ x: mouse.x, y: mouse.y, r: 0, maxR: 300, alpha: 0.55 });
    };

    const COLORS = ['#0ea5e9', '#6366f1', '#a855f7', '#ec4899', '#14b8a6', '#38bdf8'];
    const COUNT  = Math.min(Math.floor(window.innerWidth / 9), 200);

    class Particle {
      constructor() { this.reset(); }
      reset() {
        this.ox = Math.random() * canvas.width;
        this.oy = Math.random() * canvas.height;
        this.x  = this.ox; this.y = this.oy;
        this.vx = (Math.random() - 0.5) * 0.3;
        this.vy = (Math.random() - 0.5) * 0.3;
        this.r  = Math.random() * 2 + 0.8;
        this.color = COLORS[Math.floor(Math.random() * COLORS.length)];
        this.dx = 0; this.dy = 0;
      }
      applyRipple(rp) {
        const ddx = this.x - rp.x, ddy = this.y - rp.y;
        const dist = Math.sqrt(ddx * ddx + ddy * ddy);
        const band = 30;
        if (Math.abs(dist - rp.r) < band) {
          const strength = (1 - Math.abs(dist - rp.r) / band) * (1 - rp.r / rp.maxR) * 4;
          this.dx += (ddx / (dist || 1)) * strength;
          this.dy += (ddy / (dist || 1)) * strength;
        }
      }
      update() {
        this.ox += this.vx; this.oy += this.vy;
        if (this.ox < 0 || this.ox > canvas.width)  this.vx *= -1;
        if (this.oy < 0 || this.oy > canvas.height) this.vy *= -1;
        this.dx *= 0.88; this.dy *= 0.88;
        const cdx = mouse.x - this.ox, cdy = mouse.y - this.oy;
        const cdist = Math.sqrt(cdx * cdx + cdy * cdy);
        if (cdist < 120 && cdist > 0) {
          const pull = ((120 - cdist) / 120) * 0.6;
          this.dx -= (cdx / cdist) * pull;
          this.dy -= (cdy / cdist) * pull;
        }
        this.x = this.ox + this.dx;
        this.y = this.oy + this.dy;
      }
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 6; ctx.shadowColor = this.color;
        ctx.globalAlpha = 0.85;
        ctx.fill(); ctx.closePath();
      }
    }

    const particles = Array.from({ length: COUNT }, () => new Particle());

    const drawConnections = () => {
      ctx.shadowBlur = 0;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const d  = Math.sqrt(dx * dx + dy * dy);
          if (d < 120) {
            ctx.beginPath();
            ctx.globalAlpha = (1 - d / 120) * 0.35;
            ctx.strokeStyle = particles[i].color;
            ctx.lineWidth = 0.7;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke(); ctx.closePath();
          }
        }
      }
    };

    const drawRipples = () => {
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
        ctx.strokeStyle = '#06b6d4'; ctx.lineWidth = 1.5;
        ctx.globalAlpha = rp.alpha;
        ctx.shadowBlur = 12; ctx.shadowColor = '#06b6d4';
        ctx.stroke(); ctx.closePath();

        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r * 0.6, 0, Math.PI * 2);
        ctx.strokeStyle = '#a855f7'; ctx.lineWidth = 0.8;
        ctx.globalAlpha = rp.alpha * 0.5;
        ctx.stroke(); ctx.closePath();

        rp.r += 1.75; rp.alpha *= 0.96;
        if (rp.r > rp.maxR || rp.alpha < 0.01) ripples.splice(i, 1);
      }
      ctx.shadowBlur = 0;
    };

    const animate = () => {
      ctx.globalAlpha = 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (mouse.moved) { spawnRipple(); mouse.moved = false; }
      ripples.forEach(rp => particles.forEach(p => p.applyRipple(rp)));
      drawConnections();
      drawRipples();
      particles.forEach(p => { p.update(); p.draw(); });
      ctx.globalAlpha = 1;
      rafId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize',    resize);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('mouseleave', onLeave);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0, left: 0,
        width: '100%', height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        opacity: 0.75,
        mixBlendMode: 'screen',
      }}
    />
  );
};

export default ParticlesBackground;
