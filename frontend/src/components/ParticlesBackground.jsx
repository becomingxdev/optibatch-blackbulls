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

    /* ── Resize ─────────────────────────────────────────────── */
    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    /* ── Mouse / Touch tracking ─────────────────────────────── */
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

    /* ── Ripple pool ─────────────────────────────────────────── */
    const ripples = [];
    let lastRippleTime = 0;

    const spawnRipple = () => {
      if (mouse.x === -9999) return;
      const now = performance.now();
      if (now - lastRippleTime < 60) return;  // max ~16 ripples/sec
      lastRippleTime = now;
      ripples.push({ x: mouse.x, y: mouse.y, r: 0, maxR: 300, alpha: 0.55 });
    };

    /* ── Particles ───────────────────────────────────────────── */
    const COLORS = ['#0ea5e9', '#6366f1', '#a855f7', '#ec4899', '#14b8a6', '#38bdf8'];
    const COUNT  = Math.min(Math.floor(window.innerWidth / 9), 200);

    class Particle {
      constructor() { this.reset(true); }
      reset(init = false) {
        this.ox  = Math.random() * window.innerWidth;
        this.oy  = Math.random() * window.innerHeight;
        this.x   = this.ox;
        this.y   = this.oy;
        this.vx  = (Math.random() - 0.5) * 0.3;
        this.vy  = (Math.random() - 0.5) * 0.3;
        this.r   = Math.random() * 2 + 0.8;
        this.color = COLORS[Math.floor(Math.random() * COLORS.length)];
        // displacement from ripple pushes
        this.dx  = 0;
        this.dy  = 0;
      }

      applyRipple(ripple) {
        const ddx = this.x - ripple.x;
        const ddy = this.y - ripple.y;
        const dist = Math.sqrt(ddx * ddx + ddy * ddy);
        // Push particles that sit on the wavefront ±30 px
        const waveFront = ripple.r;
        const band = 30;
        if (Math.abs(dist - waveFront) < band) {
          const strength = (1 - Math.abs(dist - waveFront) / band) * (1 - ripple.r / ripple.maxR) * 4;
          const nx = ddx / (dist || 1);
          const ny = ddy / (dist || 1);
          this.dx += nx * strength;
          this.dy += ny * strength;
        }
      }

      update() {
        // Drift
        this.ox += this.vx;
        this.oy += this.vy;
        if (this.ox < 0 || this.ox > canvas.width)  this.vx *= -1;
        if (this.oy < 0 || this.oy > canvas.height) this.vy *= -1;

        // Ripple displacement decays back to 0
        this.dx *= 0.88;
        this.dy *= 0.88;

        // Cursor proximity pull (subtle)
        const cdx = mouse.x - this.ox;
        const cdy = mouse.y - this.oy;
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
        ctx.shadowBlur   = 6;
        ctx.shadowColor  = this.color;
        ctx.globalAlpha  = 0.85;
        ctx.fill();
        ctx.closePath();
      }
    }

    const particles = Array.from({ length: COUNT }, () => new Particle());

    /* ── Draw connections between nearby particles ───────────── */
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
            ctx.lineWidth   = 0.7;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
            ctx.closePath();
          }
        }
      }
    };

    /* ── Draw ripple rings ───────────────────────────────────── */
    const drawRipples = () => {
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r, 0, Math.PI * 2);
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth   = 1.5;
        ctx.globalAlpha = rp.alpha;
        ctx.shadowBlur  = 12;
        ctx.shadowColor = '#06b6d4';
        ctx.stroke();
        ctx.closePath();

        // Inner softer ring
        ctx.beginPath();
        ctx.arc(rp.x, rp.y, rp.r * 0.6, 0, Math.PI * 2);
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth   = 0.8;
        ctx.globalAlpha = rp.alpha * 0.5;
        ctx.stroke();
        ctx.closePath();

        rp.r += 1.75; rp.alpha *= 0.96;
        if (rp.r > rp.maxR || rp.alpha < 0.01) ripples.splice(i, 1);
      }
      ctx.shadowBlur = 0;
    };

    /* ── Animation loop ──────────────────────────────────────── */
    const animate = () => {
      ctx.globalAlpha = 1;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Spawn ripple on cursor move
      if (mouse.moved) { spawnRipple(); mouse.moved = false; }

      // Apply active ripples to particles
      ripples.forEach(rp => particles.forEach(p => p.applyRipple(rp)));

      // Update & draw
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
