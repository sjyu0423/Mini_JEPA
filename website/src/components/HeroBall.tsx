"use client";

import { useEffect, useRef } from "react";

/** Decorative bouncing-ball animation for the hero. Not model output. */
export function HeroBall() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let width = 0;
    let height = 0;
    let raf = 0;
    let x = 0.28;
    let y = 0.32;
    let vx = 0.42;
    let vy = 0.31;
    const radius = 0.055;
    const trail: Array<{ x: number; y: number }> = [];

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const drawFrame = () => {
      ctx.clearRect(0, 0, width, height);

      ctx.fillStyle = "#f7fafb";
      ctx.strokeStyle = "#c9d3dc";
      ctx.lineWidth = 1.5;
      roundRect(ctx, 0.5, 0.5, width - 1, height - 1, 16);
      ctx.fill();
      ctx.stroke();

      // soft grid
      ctx.strokeStyle = "rgba(18, 22, 28, 0.05)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 8; i += 1) {
        const gx = (width * i) / 8;
        const gy = (height * i) / 8;
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
        ctx.stroke();
      }

      for (let i = 0; i < trail.length; i += 1) {
        const point = trail[i];
        const alpha = ((i + 1) / trail.length) * 0.35;
        ctx.beginPath();
        ctx.fillStyle = `rgba(13, 107, 110, ${alpha})`;
        ctx.arc(point.x * width, point.y * height, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.beginPath();
      ctx.fillStyle = "#0d6b6e";
      ctx.arc(x * width, y * height, radius * Math.min(width, height), 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#5a6572";
      ctx.font = "12px IBM Plex Sans, sans-serif";
      ctx.fillText("Synthetic arena · decorative preview", 14, height - 14);
    };

    const step = () => {
      x += vx / 60;
      y += vy / 60;
      if (x < radius) {
        x = radius;
        vx = Math.abs(vx);
      } else if (x > 1 - radius) {
        x = 1 - radius;
        vx = -Math.abs(vx);
      }
      if (y < radius) {
        y = radius;
        vy = Math.abs(vy);
      } else if (y > 1 - radius) {
        y = 1 - radius;
        vy = -Math.abs(vy);
      }
      trail.push({ x, y });
      if (trail.length > 28) trail.shift();
      drawFrame();
      raf = window.requestAnimationFrame(step);
    };

    resize();
    drawFrame();
    if (!reduced) {
      raf = window.requestAnimationFrame(step);
    } else {
      trail.push({ x, y });
      drawFrame();
    }

    window.addEventListener("resize", resize);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full rounded-[16px]"
      role="img"
      aria-label="Animated bouncing ball in a square arena"
    />
  );
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}
