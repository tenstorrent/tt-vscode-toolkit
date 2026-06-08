var _TensixVizBundle = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/index.js
  var index_exports = {};
  __export(index_exports, {
    CardViz: () => CardViz,
    ClusterViz: () => ClusterViz,
    SystemViz: () => SystemViz,
    TensixViz: () => TensixViz,
    autoInit: () => autoInit
  });

  // src/chip.js
  var CHIPS = {
    wormhole: {
      label: "Wormhole (N150/N300)",
      cols: 10,
      rows: 12,
      // Actual Wormhole NOC grid: 10 cols (x=0..9) × 12 rows (y=0..11).
      // ETH:    full rows 0 and 6 (10 ETH cores each row, 20 total)
      // DRAM:   cols 0 and 5 at all non-ETH rows (y=1-5, 7-11 → 20 DRAM cells)
      // Tensix: cols 1-4 and 6-9 at all non-ETH rows (8 × 10 = 80 compute cores)
      coreType(col, row) {
        if (row === 0 || row === 6) return "eth";
        if (col === 0 || col === 5) return "dram";
        return "tensix";
      },
      // Bounding rectangle covering all Tensix/DRAM/ETH within the active grid.
      // Note: col 5 (DRAM) and row 6 (ETH) fall inside this rectangle — they
      // render with their correct core-type styling via coreType().
      computeGrid: { colStart: 1, colEnd: 9, rowStart: 1, rowEnd: 11 }
      // inclusive
    },
    blackhole: {
      label: "Blackhole (P100/P150/P300c)",
      cols: 17,
      rows: 12,
      coreType(col, row) {
        if (row === 0 || row === 11) return "dram";
        if (col === 0 || col === 16) return "eth";
        if (col === 8) return "pcie";
        if (row >= 1 && row <= 10) return "tensix";
        return "empty";
      },
      computeGrid: { colStart: 1, colEnd: 15, rowStart: 1, rowEnd: 10 }
    }
  };
  var THEME_DARK = {
    bg: "#0B1E28",
    grid: "#1A3C47",
    tensix: "#163848",
    tensixBorder: "#3A7A8C",
    tensixActive: "#4FD1C5",
    tensixPulse: "#81E6D9",
    dram: "#152035",
    dramBorder: "#2D4A6A",
    eth: "#221638",
    ethBorder: "#5B3DA0",
    pcie: "#2A2010",
    pcieBorder: "#8B6914",
    empty: "#0A1820",
    text: "#E8F0F2",
    textMuted: "#607D8B",
    teal: "#4FD1C5",
    tealLight: "#81E6D9",
    pink: "#EC96B8",
    gold: "#F4C471",
    green: "#27AE60",
    red: "#FF6B6B",
    particle: "#4FD1C5",
    heatLow: "#163848",
    heatMid: "#F4C471",
    heatHigh: "#FF6B6B",
    heatLowRgb: [22, 56, 72],
    heatMidRgb: [244, 196, 113],
    heatHighRgb: [255, 107, 107],
    nocLine: "rgba(45,102,117,0.25)",
    floatLabelBg: "rgba(13,31,45,0.88)",
    floatLabelFg: "#81E6D9"
  };
  var THEME_LIGHT = {
    bg: "#EEF4F8",
    grid: "#B8D4E0",
    tensix: "#CCDDE8",
    tensixBorder: "#6AACBE",
    tensixActive: "#0D9488",
    tensixPulse: "#0A7A70",
    dram: "#C5D8E8",
    dramBorder: "#5A9AB8",
    eth: "#C5C5E0",
    ethBorder: "#7070B8",
    pcie: "#E0D8C8",
    pcieBorder: "#A0906A",
    empty: "#E4EDF4",
    text: "#1A2C38",
    textMuted: "#4A6878",
    teal: "#0D9488",
    tealLight: "#0A7A70",
    pink: "#B01060",
    gold: "#B45309",
    green: "#15803D",
    red: "#DC2626",
    particle: "#0D9488",
    heatLow: "#CCDDE8",
    heatMid: "#D97706",
    heatHigh: "#DC2626",
    heatLowRgb: [204, 221, 232],
    heatMidRgb: [217, 119, 6],
    heatHighRgb: [220, 38, 38],
    nocLine: "rgba(70,140,160,0.35)",
    floatLabelBg: "rgba(238,244,248,0.92)",
    floatLabelFg: "#0A4A58"
  };
  var MEM_COLORS = {
    dark: {
      dram: [0, 210, 190],
      // bright teal — DRAM row glow
      l1: [255, 180, 50],
      // warm amber  — L1 fill bar
      load: [0, 210, 190],
      // teal        — DRAM→L1 read particles
      burst: [255, 200, 80],
      // gold        — prefetch / burst load
      write: [220, 80, 160]
      // vivid pink  — L1→DRAM writeback particles
    },
    light: {
      dram: [0, 140, 130],
      // deeper teal
      l1: [160, 80, 10],
      // darker amber
      load: [0, 140, 130],
      burst: [190, 130, 0],
      // darker gold
      write: [180, 40, 120]
      // deeper rose
    }
  };
  var MEM_PRESETS = {
    idle: { dram_bw: 0.05, l1_fill: 0.02, burst: false, burstHz: 0, writeback: 0, loadColor: "load" },
    inference: { dram_bw: 0.55, l1_fill: 0.45, burst: false, burstHz: 0, writeback: 0, loadColor: "load" },
    prefill: { dram_bw: 0.9, l1_fill: 0.85, burst: true, burstHz: 0.5, writeback: 0.2, loadColor: "burst" },
    thinking: { dram_bw: 0.12, l1_fill: 0.92, burst: false, burstHz: 0, writeback: 0, loadColor: "load" },
    agents: { dram_bw: 0.45, l1_fill: 0.4, burst: true, burstHz: 0.8, writeback: 0, loadColor: "load" },
    diffusion: { dram_bw: 0.65, l1_fill: 0.6, burst: true, burstHz: 1.2, writeback: 0, loadColor: "load" },
    video: { dram_bw: 0.7, l1_fill: 0.65, burst: true, burstHz: 1.6, writeback: 0, loadColor: "load" },
    batch: { dram_bw: 0.8, l1_fill: 0.6, burst: false, burstHz: 0, writeback: 0, loadColor: "load" },
    explore: { dram_bw: 0.3, l1_fill: 0.35, burst: false, burstHz: 0, writeback: 0, loadColor: "load" },
    kernel_dispatch: { dram_bw: 0.15, l1_fill: 0.55, burst: true, burstHz: "kd", writeback: 0.5, loadColor: "burst" }
  };
  function TensixViz(canvas, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.arch = opts.arch || "wormhole";
    this.speed = opts.speed || 1;
    this.chip = CHIPS[this.arch] || CHIPS.wormhole;
    this._highlights = {};
    this._particles = [];
    this._heatmap = null;
    this._labels = {};
    this._scriptQueue = [];
    this._running = false;
    this._stepMode = false;
    this._rafId = null;
    this._animGen = 0;
    this._resolveStep = null;
    this._floatLabelData = null;
    this._loop = false;
    this._loopScript = null;
    this._showMemory = !!opts.showMemory;
    this._memOverride = null;
    this._currentMode = null;
    this._cellW = 0;
    this._cellH = 0;
    this._padX = 0;
    this._padY = 0;
    this._dram = [];
    this._compute = [];
    var dpr = typeof window !== "undefined" && window.devicePixelRatio || 1;
    // Responsive sizing: cap logical dimensions to the container's actual width
    // so the canvas never overflows narrow viewports. The HTML width/height
    // attributes are the preferred size; clientWidth is the hard ceiling.
    var logicalW = canvas.width;
    var logicalH = canvas.height;
    if (typeof window !== "undefined" && canvas.parentElement) {
      var containerW = canvas.parentElement.clientWidth;
      if (containerW > 0 && containerW < logicalW) {
        logicalH = Math.round(logicalH * containerW / logicalW);
        logicalW = containerW;
      }
    }
    this._logicalW = logicalW;
    this._logicalH = logicalH;
    canvas.width = Math.round(logicalW * dpr);
    canvas.height = Math.round(logicalH * dpr);
    canvas.style.width = logicalW + "px";
    canvas.style.height = logicalH + "px";
    if (dpr > 1) {
      this.ctx.scale(dpr, dpr);
    }
    this._computeLayout();
    this.render();
  }
  TensixViz.prototype._resolveTheme = function() {
    var node = this.canvas.parentElement;
    while (node) {
      if (node.classList && node.classList.contains("tv-light")) return THEME_LIGHT;
      if (node.classList && node.classList.contains("tv-auto")) {
        return typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? THEME_LIGHT : THEME_DARK;
      }
      node = node.parentElement;
    }
    return THEME_DARK;
  };
  TensixViz.prototype._computeLayout = function() {
    const chip = this.chip;
    const pad = 8;
    const w = this._logicalW;
    const h = this._logicalH;
    this._padX = pad;
    this._padY = pad;
    this._cellW = Math.floor((w - pad * 2) / chip.cols);
    this._cellH = Math.floor((h - pad * 2) / chip.rows);
    this._dram = [];
    this._compute = [];
    for (var row = 0; row < chip.rows; row++) {
      for (var col = 0; col < chip.cols; col++) {
        var t = chip.coreType(col, row);
        if (t === "dram") this._dram.push({ col, row });
        if (t === "tensix") this._compute.push({ col, row });
      }
    }
  };
  TensixViz.prototype.render = function() {
    this._theme = this._resolveTheme();
    const ctx = this.ctx;
    const chip = this.chip;
    const lw = this._logicalW;
    const lh = this._logicalH;
    ctx.clearRect(0, 0, lw, lh);
    ctx.fillStyle = this._theme.bg;
    ctx.fillRect(0, 0, lw, lh);
    for (let row = 0; row < chip.rows; row++) {
      for (let col = 0; col < chip.cols; col++) {
        this._drawCell(col, row);
      }
    }
    if (this._heatmap) {
      this._drawHeatmap();
    }
    Object.entries(this._highlights).forEach(([key, hl]) => {
      const [col, row] = key.split(",").map(Number);
      this._drawHighlight(col, row, hl);
    });
    Object.entries(this._labels).forEach(([key, text]) => {
      const [col, row] = key.split(",").map(Number);
      this._drawCellLabel(col, row, text);
    });
    this._particles.forEach((p) => this._drawParticle(p));
    this._drawMemoryLayer();
    this._drawNocLines();
  };
  TensixViz.prototype._cellRect = function(col, row) {
    const gap = 2;
    return {
      x: this._padX + col * this._cellW + gap / 2,
      y: this._padY + row * this._cellH + gap / 2,
      w: this._cellW - gap,
      h: this._cellH - gap
    };
  };
  TensixViz.prototype._cellCenter = function(col, row) {
    const r = this._cellRect(col, row);
    return { x: r.x + r.w / 2, y: r.y + r.h / 2 };
  };
  TensixViz.prototype._drawCell = function(col, row) {
    const ctx = this.ctx;
    const type = this.chip.coreType(col, row);
    const r = this._cellRect(col, row);
    const T = this._theme;
    let fill = T[type] || T.empty;
    let border = T[type + "Border"] || fill;
    ctx.fillStyle = fill;
    this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.stroke();
    if (type !== "tensix" && type !== "empty" && r.w > 14) {
      ctx.fillStyle = T.textMuted;
      ctx.font = `bold ${Math.max(7, r.w * 0.3)}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const labels = { dram: "D", eth: "E", pcie: "P" };
      if (labels[type]) ctx.fillText(labels[type], r.x + r.w / 2, r.y + r.h / 2);
    }
  };
  TensixViz.prototype._drawNocLines = function() {
    const ctx = this.ctx;
    const chip = this.chip;
    const cg = chip.computeGrid;
    ctx.strokeStyle = this._theme.nocLine;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([2, 4]);
    for (let row = cg.rowStart; row <= cg.rowEnd; row++) {
      const y = this._padY + row * this._cellH + this._cellH / 2;
      const x0 = this._padX + cg.colStart * this._cellW;
      const x1 = this._padX + (cg.colEnd + 1) * this._cellW;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
    }
    for (let col = cg.colStart; col <= cg.colEnd; col++) {
      const x = this._padX + col * this._cellW + this._cellW / 2;
      const y0 = this._padY + cg.rowStart * this._cellH;
      const y1 = this._padY + (cg.rowEnd + 1) * this._cellH;
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  };
  TensixViz.prototype._drawMemoryLayer = function() {
    if (!this._showMemory || !this._memPhase) return;
    const ctx = this.ctx;
    const chip = this.chip;
    const cg = chip.computeGrid;
    const T = this._theme;
    const isDark = T === THEME_DARK;
    const mc = isDark ? MEM_COLORS.dark : MEM_COLORS.light;
    const mode = this._currentMode || "idle";
    const preset = MEM_PRESETS[mode] || MEM_PRESETS.idle;
    const mem = this._memPhase;
    let env;
    if (preset.burstHz === "kd") {
      env = mem.kdGlow;
    } else if (preset.burst) {
      env = 0.5 + 0.5 * Math.cos(mem.burstPhase * Math.PI * 2);
    } else {
      env = 0.85 + 0.15 * Math.sin(mem.phase * Math.PI * 2);
    }
    const dramBw = this._memOverride && this._memOverride.dram_bw !== void 0 ? this._memOverride.dram_bw : preset.dram_bw;
    const l1Fill = this._memOverride && this._memOverride.l1_fill !== void 0 ? this._memOverride.l1_fill : preset.l1_fill;
    const dramAlpha = dramBw * env * 0.55;
    if (dramAlpha > 5e-3) {
      const dramColor = mc.dram;
      ctx.save();
      ctx.globalAlpha = dramAlpha;
      ctx.fillStyle = "rgb(" + dramColor[0] + "," + dramColor[1] + "," + dramColor[2] + ")";
      for (let di = 0; di < this._dram.length; di++) {
        const dc = this._dram[di];
        const r = this._cellRect(dc.col, dc.row);
        this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
        ctx.fill();
      }
      ctx.restore();
    }
    const spawnRate = dramBw * env;
    if (spawnRate > 0 && this._compute.length > 0 && this._dram.length > 0) {
      if (Math.random() < spawnRate * 0.3) {
        const fromCell = this._dram[Math.floor(Math.random() * this._dram.length)];
        const toCell = this._compute[Math.floor(Math.random() * this._compute.length)];
        const fromRect = this._cellRect(fromCell.col, fromCell.row);
        const toRect = this._cellRect(toCell.col, toCell.row);
        const loadColorArr = mc[preset.loadColor] || mc.load;
        this._particles.push({
          x: fromRect.x + fromRect.w / 2,
          y: fromRect.y + fromRect.h / 2,
          startX: fromRect.x + fromRect.w / 2,
          startY: fromRect.y + fromRect.h / 2,
          toX: toRect.x + toRect.w / 2,
          toY: toRect.y + toRect.h / 2,
          progress: 0,
          speed: 8e-3 + Math.random() * 0.012,
          color: "rgb(" + loadColorArr[0] + "," + loadColorArr[1] + "," + loadColorArr[2] + ")",
          radius: 1.5,
          alpha: 0.8,
          _isMem: true
        });
      }
      if (preset.writeback > 0 && Math.random() < preset.writeback * spawnRate * 0.15) {
        const wbFrom = this._compute[Math.floor(Math.random() * this._compute.length)];
        const wbTo = this._dram[Math.floor(Math.random() * this._dram.length)];
        const wbFromRect = this._cellRect(wbFrom.col, wbFrom.row);
        const wbToRect = this._cellRect(wbTo.col, wbTo.row);
        const writeColorArr = mc.write;
        this._particles.push({
          x: wbFromRect.x + wbFromRect.w / 2,
          y: wbFromRect.y + wbFromRect.h / 2,
          startX: wbFromRect.x + wbFromRect.w / 2,
          startY: wbFromRect.y + wbFromRect.h / 2,
          toX: wbToRect.x + wbToRect.w / 2,
          toY: wbToRect.y + wbToRect.h / 2,
          progress: 0,
          speed: 8e-3 + Math.random() * 0.012,
          color: "rgb(" + writeColorArr[0] + "," + writeColorArr[1] + "," + writeColorArr[2] + ")",
          radius: 1.5,
          alpha: 0.8,
          _isMem: true
        });
      }
    }
    for (let i = this._particles.length - 1; i >= 0; i--) {
      const p = this._particles[i];
      if (!p._isMem) continue;
      p.progress = Math.min(1, p.progress + p.speed);
      p.x = p.startX + (p.toX - p.startX) * p.progress;
      p.y = p.startY + (p.toY - p.startY) * p.progress;
      if (p.progress >= 1) {
        this._particles.splice(i, 1);
      }
    }
    if (l1Fill > 0.01) {
      const lc = mc.l1;
      ctx.save();
      ctx.globalAlpha = 0.75;
      ctx.fillStyle = "rgb(" + lc[0] + "," + lc[1] + "," + lc[2] + ")";
      for (let lrow = cg.rowStart; lrow <= cg.rowEnd; lrow++) {
        for (let lcol = cg.colStart; lcol <= cg.colEnd; lcol++) {
          if (chip.coreType(lcol, lrow) !== "tensix") continue;
          const cr = this._cellRect(lcol, lrow);
          const noise = 1 + 0.15 * Math.sin(lcol * 3.7 + lrow * 2.9 + mem.phase * Math.PI);
          const fillH = Math.min(cr.h * 0.9, cr.h * l1Fill * noise);
          const barW = cr.w * 0.7;
          const barX = cr.x + (cr.w - barW) / 2;
          const barY = cr.y + cr.h - fillH;
          ctx.fillRect(barX, barY, barW, fillH);
        }
      }
      ctx.restore();
    }
  };
  TensixViz.prototype._drawHighlight = function(col, row, hl) {
    const ctx = this.ctx;
    const r = this._cellRect(col, row);
    const alpha = hl.alpha !== void 0 ? hl.alpha : 1;
    const T = this._theme;
    ctx.save();
    ctx.globalAlpha = alpha;
    const color = T[hl.color] || hl.color || T.tensixActive;
    const bright = hl.color === "pink" ? T.pink : T.tensixPulse;
    ctx.fillStyle = color;
    this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.fill();
    ctx.strokeStyle = bright;
    ctx.lineWidth = 1.5;
    this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.stroke();
    ctx.restore();
  };
  TensixViz.prototype._drawCellLabel = function(col, row, text) {
    const ctx = this.ctx;
    const c = this._cellCenter(col, row);
    const r = this._cellRect(col, row);
    ctx.save();
    ctx.font = `bold ${Math.max(7, r.w * 0.28)}px sans-serif`;
    ctx.fillStyle = this._theme.text;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0,0,0,0.8)";
    ctx.shadowBlur = 4;
    ctx.fillText(text.length > 5 ? text.slice(0, 4) + "\u2026" : text, c.x, c.y);
    ctx.restore();
  };
  TensixViz.prototype._drawParticle = function(p) {
    const ctx = this.ctx;
    const color = p.color || this._theme.particle;
    const r = p.radius || 4;
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };
  TensixViz.prototype._drawHeatmap = function() {
    const ctx = this.ctx;
    const chip = this.chip;
    const cg = chip.computeGrid;
    let maxVal = 0;
    for (let row = cg.rowStart; row <= cg.rowEnd; row++) {
      for (let col = cg.colStart; col <= cg.colEnd; col++) {
        if (chip.coreType(col, row) !== "tensix") continue;
        const v = (this._heatmap[row] || [])[col] || 0;
        if (v > maxVal) maxVal = v;
      }
    }
    if (maxVal === 0) return;
    for (let row = cg.rowStart; row <= cg.rowEnd; row++) {
      for (let col = cg.colStart; col <= cg.colEnd; col++) {
        if (chip.coreType(col, row) !== "tensix") continue;
        const v = ((this._heatmap[row] || [])[col] || 0) / maxVal;
        const r = this._cellRect(col, row);
        const color = this._heatColor(v, this._theme);
        ctx.save();
        ctx.globalAlpha = 0.6;
        ctx.fillStyle = color;
        this._roundRect(ctx, r.x, r.y, r.w, r.h, 3);
        ctx.fill();
        ctx.restore();
      }
    }
  };
  TensixViz.prototype._heatColor = function(t, T) {
    function lerp(a, b, t2) {
      return Math.round(a + (b - a) * t2);
    }
    function hex(r, g, b) {
      return "rgb(" + r + "," + g + "," + b + ")";
    }
    const low = T.heatLowRgb;
    const mid = T.heatMidRgb;
    const high = T.heatHighRgb;
    if (t < 0.5) {
      const s = t * 2;
      return hex(lerp(low[0], mid[0], s), lerp(low[1], mid[1], s), lerp(low[2], mid[2], s));
    } else {
      const s = (t - 0.5) * 2;
      return hex(lerp(mid[0], high[0], s), lerp(mid[1], high[1], s), lerp(mid[2], high[2], s));
    }
  };
  TensixViz.prototype._roundRect = function(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  };
  TensixViz.prototype.renderLegend = function(legendEl) {
    const T = this._resolveTheme();
    const items = [
      { color: T.tensixActive, label: "Active compute core" },
      { color: T.tensix, label: "Idle Tensix core" },
      { color: T.dramBorder, label: "DRAM controller" },
      { color: T.ethBorder, label: "Ethernet link" },
      { color: T.particle, label: "Data tile (NOC transfer)" }
    ];
    legendEl.replaceChildren ? legendEl.replaceChildren() : legendEl.innerHTML = "";
    items.forEach(function(item) {
      var span = document.createElement("span");
      span.className = "tv-legend-item";
      var dot = document.createElement("span");
      dot.className = "tv-legend-dot";
      dot.style.background = item.color;
      span.appendChild(dot);
      span.appendChild(document.createTextNode(" " + item.label));
      legendEl.appendChild(span);
    });
  };
  TensixViz.prototype.play = function(script) {
    this._loopScript = script;
    this._stepMode = false;
    this._scriptQueue = script.slice();
    if (!this._running) this._runLoop();
    return this;
  };
  TensixViz.prototype.stepThrough = function(script) {
    this._stepMode = true;
    this._scriptQueue = script.slice();
    this._running = false;
    return this;
  };
  TensixViz.prototype.next = function() {
    if (this._resolveStep) {
      const fn = this._resolveStep;
      this._resolveStep = null;
      fn();
    } else if (this._scriptQueue.length > 0) {
      this._runNextStep();
    }
  };
  TensixViz.prototype.reset = function() {
    this._animGen++;
    this._running = false;
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this._highlights = {};
    this._particles = [];
    this._heatmap = null;
    this._labels = {};
    this._floatLabelData = null;
    this._memOverride = null;
    this._memPhase = null;
    this._currentMode = null;
    this._scriptQueue = [];
    this._resolveStep = null;
    this.render();
  };
  TensixViz.prototype.setMemoryStats = function(stats) {
    if (!stats || typeof stats !== "object") return;
    this._memOverride = {};
    if (typeof stats.dram_bw === "number") this._memOverride.dram_bw = Math.max(0, Math.min(1, stats.dram_bw));
    if (typeof stats.l1_fill === "number") this._memOverride.l1_fill = Math.max(0, Math.min(1, stats.l1_fill));
  };
  TensixViz.prototype._runLoop = function() {
    const self = this;
    self._running = true;
    const gen = self._animGen;
    function next() {
      if (!self._running || self._animGen !== gen) return;
      if (self._scriptQueue.length === 0) {
        if (self._loop && self._loopScript) {
          setTimeout(function() {
            if (!self._running || self._animGen !== gen) return;
            self._highlights = {};
            self._particles = [];
            self._heatmap = null;
            self._labels = {};
            self._floatLabelData = null;
            self.render();
            self._scriptQueue = self._loopScript.slice();
            next();
          }, 600);
          return;
        }
        self._running = false;
        return;
      }
      const step = self._scriptQueue.shift();
      self._execStep(step, function() {
        if (self._stepMode) {
          self._resolveStep = next;
        } else {
          next();
        }
      });
    }
    next();
  };
  TensixViz.prototype._runNextStep = function() {
    const self = this;
    if (self._scriptQueue.length === 0) return;
    const step = self._scriptQueue.shift();
    self._execStep(step, function() {
      self._resolveStep = self._runNextStep.bind(self);
    });
  };
  TensixViz.prototype._execStep = function(step, done) {
    const self = this;
    switch (step.step) {
      case "highlight":
        return self._stepHighlight(step, done);
      case "unhighlight":
        return self._stepUnhighlight(step, done);
      case "transfer":
        return self._stepTransfer(step, done);
      case "heatmap":
        return self._stepHeatmap(step, done);
      case "label":
        return self._stepLabel(step, done);
      case "clear":
        return self._stepClear(step, done);
      case "pause":
        return self._stepPause(step, done);
      default:
        done();
        break;
    }
  };
  TensixViz.prototype._stepHighlight = function(step, done) {
    const self = this;
    const cores = step.cores || [];
    const color = step.color || "tensixActive";
    const label = step.label || "";
    const ms = (step.ms || 600) / self.speed;
    const gen = self._animGen;
    const start = performance.now();
    const half = ms / 2;
    cores.forEach(([col, row]) => {
      self._highlights[col + "," + row] = { color, alpha: 0, label };
    });
    if (label) self._floatLabel(cores, label);
    function tick(now) {
      if (self._animGen !== gen) return;
      const elapsed = now - start;
      const alpha = elapsed < half ? elapsed / half : 1;
      cores.forEach(([col, row]) => {
        if (self._highlights[col + "," + row]) {
          self._highlights[col + "," + row].alpha = alpha;
        }
      });
      self.render();
      if (elapsed < ms) {
        self._rafId = requestAnimationFrame(tick);
      } else {
        cores.forEach(([col, row]) => {
          if (self._highlights[col + "," + row]) {
            self._highlights[col + "," + row].alpha = 1;
          }
        });
        self.render();
        done();
      }
    }
    self._rafId = requestAnimationFrame(tick);
  };
  TensixViz.prototype._stepUnhighlight = function(step, done) {
    const self = this;
    const cores = step.cores ? step.cores : Object.keys(this._highlights).map(function(k) {
      return k.split(",").map(Number);
    });
    const ms = (step.ms !== void 0 ? step.ms : 250) / this.speed;
    this._floatLabelData = null;
    if (ms <= 0 || cores.length === 0) {
      cores.forEach(function(cr) {
        delete self._highlights[cr[0] + "," + cr[1]];
      });
      this.render();
      done();
      return;
    }
    const initial = {};
    cores.forEach(function(cr) {
      const key = cr[0] + "," + cr[1];
      initial[key] = self._highlights[key] ? self._highlights[key].alpha || 1 : 1;
    });
    const start = performance.now();
    const gen = self._animGen;
    function tick(now) {
      if (self._animGen !== gen) return;
      const t = Math.min(1, (now - start) / ms);
      cores.forEach(function(cr) {
        const key = cr[0] + "," + cr[1];
        if (self._highlights[key]) {
          self._highlights[key].alpha = (initial[key] || 1) * (1 - t);
        }
      });
      self.render();
      if (t < 1) {
        self._rafId = requestAnimationFrame(tick);
      } else {
        cores.forEach(function(cr) {
          delete self._highlights[cr[0] + "," + cr[1]];
        });
        self.render();
        done();
      }
    }
    self._rafId = requestAnimationFrame(tick);
  };
  TensixViz.prototype._stepTransfer = function(step, done) {
    const self = this;
    const from = step.from;
    const to = step.to;
    const T = this._resolveTheme();
    const color = T[step.color] || T.particle;
    const ms = (step.ms || 800) / self.speed;
    const start = performance.now();
    const p0 = self._cellCenter(from[0], from[1]);
    const p1 = self._cellCenter(to[0], to[1]);
    const mid = { x: p1.x, y: p0.y };
    const particle = { x: p0.x, y: p0.y, color, radius: Math.max(3, self._cellW * 0.15) };
    self._particles.push(particle);
    const gen = self._animGen;
    function tick(now) {
      if (self._animGen !== gen) return;
      const t = Math.min(1, (now - start) / ms);
      const e = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      if (e < 0.5) {
        const s = e * 2;
        particle.x = p0.x + (mid.x - p0.x) * s;
        particle.y = p0.y;
      } else {
        const s = (e - 0.5) * 2;
        particle.x = mid.x;
        particle.y = p0.y + (p1.y - p0.y) * s;
      }
      self.render();
      if (t < 1) {
        self._rafId = requestAnimationFrame(tick);
      } else {
        const idx = self._particles.indexOf(particle);
        if (idx !== -1) self._particles.splice(idx, 1);
        self.render();
        done();
      }
    }
    self._rafId = requestAnimationFrame(tick);
  };
  TensixViz.prototype._stepHeatmap = function(step, done) {
    const self = this;
    const gen = self._animGen;
    self._heatmap = step.data || null;
    self.render();
    setTimeout(function() {
      if (self._animGen === gen) done();
    }, (step.ms || 200) / self.speed);
  };
  TensixViz.prototype._stepLabel = function(step, done) {
    const core = step.core;
    if (core) this._labels[core[0] + "," + core[1]] = step.text || "";
    this.render();
    done();
  };
  TensixViz.prototype._stepClear = function(step, done) {
    if (step.what === "highlights" || !step.what) {
      this._highlights = {};
      this._floatLabelData = null;
    }
    if (step.what === "labels" || !step.what) this._labels = {};
    if (step.what === "heatmap" || !step.what) this._heatmap = null;
    if (step.what === "particles" || !step.what) this._particles = [];
    this.render();
    done();
  };
  TensixViz.prototype._stepPause = function(step, done) {
    const self = this;
    const gen = self._animGen;
    setTimeout(function() {
      if (self._animGen === gen) done();
    }, (step.ms || 500) / self.speed);
  };
  TensixViz.prototype.activate = function(mode, opts) {
    this.reset();
    this._currentMode = mode;
    opts = opts || {};
    var chip = this.chip;
    var cg = chip.computeGrid;
    var W = cg.colEnd - cg.colStart + 1;
    var H = cg.rowEnd - cg.rowStart + 1;
    var t = opts.phaseOffset || 0;
    var gen = this._animGen;
    var self = this;
    var prev = [];
    for (var r = 0; r < H; r++) {
      prev[r] = [];
      for (var c = 0; c < W; c++) prev[r][c] = 0;
    }
    var _kd = {
      list: [{
        c: Math.floor((W - 4) / 2),
        // center a 4×3 seed kernel
        r: Math.floor((H - 3) / 2),
        w: 4,
        h: 3,
        age: 12,
        // already propagated, mid-life bright
        maxAge: 40,
        seed: 37
      }],
      nextDispatch: 1
      // dispatch another kernel on the first frame
    };
    var _mem = {
      phase: 0,
      // continuously incrementing phase counter
      burstPhase: 0,
      // fractional phase within a burst cycle (0–1)
      kdGlow: 0
      // current DRAM glow for kernel_dispatch (decays per frame)
    };
    var MODES = {
      idle: function(c2, r2) {
        return Math.min(1, prev[r2][c2] * 0.9 + (Math.random() < 0.03 ? Math.random() * 0.35 : 0));
      },
      inference: function(c2, r2) {
        var wave = t % 1 * W;
        return Math.max(0, 1 - Math.abs(c2 - wave) / 3) * 0.9;
      },
      diffusion: function(c2, r2) {
        var cx = W / 2, cy = H / 2;
        var dist = Math.sqrt((c2 - cx) * (c2 - cx) + (r2 - cy) * (r2 - cy));
        var ring = t % 1 * Math.sqrt(cx * cx + cy * cy);
        return Math.max(0, 1 - Math.abs(dist - ring) / 2) * 0.9;
      },
      agents: function(c2, r2) {
        return Math.min(1, prev[r2][c2] * 0.85 + (Math.random() < 0.06 ? Math.random() * 0.8 : 0));
      },
      explore: function(c2, r2) {
        return (Math.sin(c2 * 0.6 + t * Math.PI * 4) * Math.cos(r2 * 0.4 + t * Math.PI * 2) + 1) / 2 * 0.85;
      },
      // ── LLM-specific states ──────────────────────────────────────────────────
      thinking: function(c2, r2) {
        return (Math.sin(t * Math.PI * 0.7 + c2 * 0.18 + r2 * 0.12) + 1) / 2 * 0.4 + 0.45;
      },
      prefill: function(c2, r2) {
        var wave = t * 1.5 % 1 * (W + 6) - 3;
        return Math.max(0, 1 - Math.abs(c2 - wave) / (W * 0.5)) * 0.95;
      },
      video: function(c2, r2) {
        var cx = W / 2, cy = H / 2;
        var dist = Math.sqrt((c2 - cx) * (c2 - cx) + (r2 - cy) * (r2 - cy));
        var maxR = Math.sqrt(cx * cx + cy * cy);
        var r1 = Math.max(0, 1 - Math.abs(dist - t % 1 * maxR) / 1.8) * 0.9;
        var r22 = Math.max(0, 1 - Math.abs(dist - (t + 0.5) % 1 * maxR) / 1.8) * 0.9;
        return Math.max(r1, r22);
      },
      batch: function(c2, r2) {
        var speed = 0.7;
        var w1 = Math.max(0, 1 - Math.abs(c2 - t * speed % 1 * W) / 2) * 0.85;
        var w2 = Math.max(0, 1 - Math.abs(c2 - (t * speed + 0.33) % 1 * W) / 2) * 0.85;
        var w3 = Math.max(0, 1 - Math.abs(c2 - (t * speed + 0.66) % 1 * W) / 2) * 0.85;
        return Math.max(w1, w2, w3);
      },
      kernel_dispatch: function(c2, r2) {
        if (c2 === 0 && r2 === 0) {
          for (var i = _kd.list.length - 1; i >= 0; i--) {
            _kd.list[i].age++;
            if (_kd.list[i].age >= _kd.list[i].maxAge) _kd.list.splice(i, 1);
          }
          if (--_kd.nextDispatch <= 0) {
            var kw = 1 + Math.floor(Math.random() * Math.min(8, W - 1));
            var kh = 1 + Math.floor(Math.random() * Math.min(6, H - 1));
            var kc = Math.floor(Math.random() * (W - kw));
            var kr = Math.floor(Math.random() * (H - kh));
            _kd.list.push({
              c: kc,
              r: kr,
              w: kw,
              h: kh,
              age: 0,
              maxAge: 38 + Math.floor(Math.random() * 50),
              seed: Math.random() * 100
            });
            if (Math.random() < 0.4) {
              kw = 1 + Math.floor(Math.random() * Math.min(6, W - 1));
              kh = 1 + Math.floor(Math.random() * Math.min(4, H - 1));
              kc = Math.floor(Math.random() * (W - kw));
              kr = Math.floor(Math.random() * (H - kh));
              _kd.list.push({
                c: kc,
                r: kr,
                w: kw,
                h: kh,
                age: 0,
                maxAge: 38 + Math.floor(Math.random() * 50),
                seed: Math.random() * 100
              });
            }
            _kd.nextDispatch = 18 + Math.floor(Math.random() * 28);
          }
        }
        var val = 0;
        for (var i = 0; i < _kd.list.length; i++) {
          var k = _kd.list[i];
          if (c2 < k.c || c2 >= k.c + k.w || r2 < k.r || r2 >= k.r + k.h) continue;
          var dist = c2 - k.c + (r2 - k.r);
          var effectiveAge = k.age - dist * 1.5;
          if (effectiveAge < 0) continue;
          var fadeIn = Math.min(1, effectiveAge / 4);
          var fadeOut = Math.min(1, (k.maxAge - k.age) / 10);
          var noise = 0.72 + 0.28 * Math.sin(c2 * 7.3 + r2 * 4.1 + k.seed + t * 6);
          val = Math.max(val, fadeIn * fadeOut * noise * 0.88);
        }
        return val;
      }
    };
    var fn = MODES[mode];
    if (!fn) throw new Error('Unknown animation mode: "' + mode + '"');
    self._running = true;
    if (self._showMemory) {
      self._memPhase = _mem;
    }
    function tick() {
      if (self._animGen !== gen) return;
      t += 0.012;
      if (self._showMemory) {
        _mem.phase += 0.012;
        var preset = MEM_PRESETS[mode] || MEM_PRESETS.idle;
        if (preset.burst && preset.burstHz !== "kd") {
          _mem.burstPhase = (_mem.burstPhase + preset.burstHz * 0.012) % 1;
        }
        if (preset.burstHz === "kd") {
          var kdActive = _kd.list.length > 0 ? 1 : 0;
          _mem.kdGlow = _mem.kdGlow * 0.92 + kdActive * 0.08;
        }
        self._memPhase = _mem;
      }
      var next = [];
      var hmap = [];
      for (var r2 = 0; r2 < H; r2++) {
        next[r2] = [];
        var row = r2 + cg.rowStart;
        if (!hmap[row]) hmap[row] = [];
        for (var c2 = 0; c2 < W; c2++) {
          var col = c2 + cg.colStart;
          var v = fn(c2, r2);
          next[r2][c2] = v;
          hmap[row][col] = v;
        }
      }
      prev = next;
      self._heatmap = hmap;
      self.render();
      self._rafId = requestAnimationFrame(tick);
    }
    self._rafId = requestAnimationFrame(tick);
  };
  TensixViz.prototype._floatLabel = function(cores, text) {
    if (!cores.length) return;
    const self = this;
    const ctx = this.ctx;
    let minCol = Infinity, maxCol = -Infinity, minRow = Infinity;
    cores.forEach(([col, row]) => {
      minCol = Math.min(minCol, col);
      maxCol = Math.max(maxCol, col);
      minRow = Math.min(minRow, row);
    });
    const leftCenter = self._cellCenter(minCol, minRow);
    const rightCenter = self._cellCenter(maxCol, minRow);
    const cx = (leftCenter.x + rightCenter.x) / 2;
    const cy = leftCenter.y - self._cellH * 0.8;
    this._floatLabelData = { cx, cy, text };
  };
  var _origRender = TensixViz.prototype.render;
  TensixViz.prototype.render = function() {
    _origRender.call(this);
    if (this._floatLabelData) {
      const ctx = this.ctx;
      const { cx: rawCx, cy: rawCy, text } = this._floatLabelData;
      const pad = 6;
      ctx.font = "bold 11px sans-serif";
      const w = ctx.measureText(text).width + pad * 2;
      const h = 18;
      // Clamp so the label box never overflows any canvas edge.
      const margin = 4;
      const cx = Math.max(w / 2 + margin, Math.min(this._logicalW - w / 2 - margin, rawCx));
      const cy = Math.max(h / 2 + margin, Math.min(this._logicalH - h / 2 - margin, rawCy));
      const T = this._theme;
      ctx.fillStyle = T.floatLabelBg;
      ctx.strokeStyle = T.teal;
      ctx.lineWidth = 1;
      this._roundRect(ctx, cx - w / 2, cy - h / 2, w, h, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = T.floatLabelFg;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, cx, cy);
    }
  };
  TensixViz.makeParallelismScript = function(totalCores, serialFraction) {
    serialFraction = serialFraction || 0.1;
    const steps = [];
    const allCores = [];
    for (let row = 1; row <= 8; row++) {
      for (let col = 1; col <= 8; col++) {
        allCores.push([col, row]);
        if (allCores.length >= totalCores) break;
      }
      if (allCores.length >= totalCores) break;
    }
    steps.push({ step: "highlight", cores: [allCores[0]], color: "pink", label: "1 core (serial)" });
    steps.push({ step: "pause", ms: 900 });
    for (let n = 2; n <= totalCores; n += Math.ceil(totalCores / 5)) {
      const active = allCores.slice(0, n);
      const speedup = (1 / (serialFraction + (1 - serialFraction) / n)).toFixed(1);
      steps.push({ step: "unhighlight" });
      steps.push({ step: "highlight", cores: active, color: "tensixActive", label: n + " cores \xB7 " + speedup + "x" });
      steps.push({ step: "pause", ms: 700 });
    }
    return steps;
  };
  TensixViz.makeNocScript = function(src, dst) {
    src = src || [1, 1];
    dst = dst || [7, 7];
    return [
      { step: "highlight", cores: [src], color: "teal", label: "src" },
      { step: "highlight", cores: [dst], color: "pink", label: "dst" },
      { step: "pause", ms: 400 },
      { step: "transfer", from: src, to: dst, ms: 1e3 },
      { step: "pause", ms: 300 }
    ];
  };
  TensixViz.autoInit = function() {
    document.querySelectorAll(".tensix-viz-container").forEach(function(container) {
      const canvas = container.querySelector(".tensix-viz-canvas");
      const playBtn = container.querySelector(".tv-play");
      const stepBtn = container.querySelector(".tv-step");
      const legendEl = container.querySelector(".tv-legend");
      if (!canvas) return;
      const arch = container.dataset.arch || "wormhole";
      const viz = new TensixViz(canvas, { arch });
      let script = [];
      try {
        script = JSON.parse(container.dataset.script || "[]");
      } catch (e) {
      }
      if (legendEl) viz.renderLegend(legendEl);
      var startViz = function() {
        viz._loop = true;
        viz.play(script);
      };
      if (typeof requestIdleCallback !== "undefined") {
        requestIdleCallback(startViz, { timeout: 3e3 });
      } else {
        setTimeout(startViz, 1500);
      }
      if (playBtn) {
        playBtn.addEventListener("click", function() {
          const icon = playBtn.textContent.trim();
          if (icon.startsWith("\u25B6")) {
            playBtn.textContent = "\u23F8";
            viz.reset();
            viz.play(script);
          } else {
            playBtn.textContent = "\u25B6";
            viz.reset();
          }
        });
      }
      if (stepBtn) {
        stepBtn.addEventListener("click", function() {
          if (!viz._stepMode) {
            viz.stepThrough(script);
          }
          viz.next();
        });
      }
    });
  };

  // topologies/bh-chip.json
  var bh_chip_default = {
    arch: "blackhole",
    cols: 17,
    rows: 12,
    dram_rows: [0, 11],
    eth_cols: [0, 16],
    pcie_col: 8,
    compute_grid: { col_start: 1, col_end: 15, row_start: 1, row_end: 10 },
    eth_links: { per_side: 2, sides: ["N", "S", "E", "W"] }
  };

  // topologies/wh-chip.json
  var wh_chip_default = {
    arch: "wormhole",
    cols: 10,
    rows: 12,
    dram_rows: [0, 9],
    eth_cols: [0, 9],
    compute_grid: { col_start: 1, col_end: 8, row_start: 1, row_end: 8 },
    eth_links: { per_side: 4, sides: ["N", "S", "E", "W"] }
  };

  // topologies/bh-p300c.json
  var bh_p300c_default = {
    chips: ["bh-chip", "bh-chip"],
    layout: "horizontal",
    labels: ["ASIC1 (left)", "ASIC0 (right)"],
    intra_links: [
      { from: { chip: 0, eth: [10, 11] }, to: { chip: 1, eth: [2, 3] } }
    ]
  };

  // topologies/wh-n300.json
  var wh_n300_default = {
    chips: ["wh-chip", "wh-chip"],
    layout: "horizontal",
    labels: ["Left (L)", "Right (R)"],
    intra_links: [
      { from: { chip: 0 }, to: { chip: 1 }, link_count: 2, speed_gbps: 100 }
    ]
  };

  // topologies/qb2.json
  var qb2_default = {
    cards: ["bh-p300c", "bh-p300c"],
    layout: "vertical",
    labels: ["CARD 0", "CARD 1"],
    inter_links: [
      { connector: "samtec", label: "Samtec cable \xB7 inter-card link", from: { card: 0 }, to: { card: 1 } }
    ]
  };

  // topologies/t3000.json
  var t3000_default = {
    cards: ["wh-n300", "wh-n300", "wh-n300", "wh-n300"],
    layout: "grid",
    grid: [2, 4],
    labels: ["Card 0", "Card 1", "Card 2", "Card 3"],
    inter_links: [
      { topology: "2d_mesh", link_count: 2, speed_gbps: 100 }
    ]
  };

  // topologies/bh-galaxy.json
  var bh_galaxy_default = {
    chip_count: 32,
    grid: [4, 8],
    arch: "blackhole",
    mesh_links: "full_2d",
    eth_ports: 56,
    eth_speed: "800G"
  };

  // topologies/bh-galaxy-sc.json
  var bh_galaxy_sc_default = {
    servers: 4,
    server_config: "bh-galaxy",
    topology: "2d_torus",
    grid: [4, 32],
    total_chips: 128
  };

  // src/topology.js
  var REGISTRY = {
    "bh-chip": bh_chip_default,
    "wh-chip": wh_chip_default,
    "bh-p300c": bh_p300c_default,
    "wh-n300": wh_n300_default,
    "qb2": qb2_default,
    "t3000": t3000_default,
    "bh-galaxy": bh_galaxy_default,
    "bh-galaxy-sc": bh_galaxy_sc_default
  };
  function loadTopology(nameOrConfig) {
    if (typeof nameOrConfig === "string") {
      const topo = REGISTRY[nameOrConfig];
      if (!topo) {
        throw new Error(
          `Unknown topology: "${nameOrConfig}". Available: ${Object.keys(REGISTRY).join(", ")}`
        );
      }
      return topo;
    }
    if (nameOrConfig === null || typeof nameOrConfig !== "object" || Array.isArray(nameOrConfig)) {
      throw new TypeError(
        `loadTopology: expected a topology name (string) or config object, got ${JSON.stringify(nameOrConfig)}`
      );
    }
    return nameOrConfig;
  }

  // src/card.js
  function CardViz(container, config, options) {
    this._container = container;
    this._topo = loadTopology(config);
    this._options = options || {};
    this._chips = [];
    this._chipEls = [];
    this._zoomed = -1;
    this._breadcrumb = null;
    this._destroyed = false;
    this._init();
  }
  CardViz.prototype._init = function() {
    const self = this;
    const topo = this._topo;
    const container = this._container;
    container.classList.add("tv-card");
    topo.chips.forEach(function(chipName, i) {
      const wrapper = document.createElement("div");
      wrapper.classList.add("tv-chip-wrapper");
      wrapper.dataset.chipIndex = String(i);
      const label = document.createElement("div");
      label.classList.add("tv-chip-label");
      label.textContent = topo.labels && topo.labels[i] || "Chip " + i;
      wrapper.appendChild(label);
      const canvas = document.createElement("canvas");
      canvas.width = self._options.chipWidth || 340;
      canvas.height = self._options.chipHeight || 240;
      wrapper.appendChild(canvas);
      container.appendChild(wrapper);
      self._chipEls.push(wrapper);
      const chipTopo = loadTopology(chipName);
      const viz = new TensixViz(canvas, { arch: chipTopo.arch });
      self._chips.push(viz);
      if (i < topo.chips.length - 1) {
        const link = document.createElement("div");
        link.classList.add("tv-card-link");
        link.title = "Intra-card ETH link";
        container.appendChild(link);
      }
    });
  };
  CardViz.prototype.activate = function(mode) {
    if (this._destroyed) return;
    this._chips.forEach(function(chip, i) {
      setTimeout(function() {
        chip.activate(mode);
      }, i * 120);
    });
  };
  CardViz.prototype.reset = function() {
    if (this._destroyed) return;
    this._chips.forEach(function(chip) {
      chip.reset();
    });
  };
  CardViz.prototype.highlight = function(indices) {
    if (this._destroyed) return;
    const self = this;
    this._chipEls.forEach(function(el, i) {
      if (indices.indexOf(i) !== -1) {
        el.classList.add("tv-highlighted");
      } else {
        el.classList.remove("tv-highlighted");
      }
    });
  };
  CardViz.prototype.transitionTo = function(level, opts) {
    if (this._destroyed) return Promise.resolve();
    const self = this;
    opts = opts || {};
    if (level === "chip") {
      const idx = opts.index != null ? opts.index : 0;
      this._zoomed = idx;
      this._chipEls.forEach(function(el, i) {
        if (i === idx) {
          el.classList.add("tv-active");
          el.classList.remove("tv-hidden");
        } else {
          el.classList.remove("tv-active");
          el.classList.add("tv-hidden");
        }
      });
      this._container.classList.add("tv-zoomed-in");
      this._showBreadcrumb("Card \u203A Chip " + idx);
    } else {
      this._zoomed = -1;
      this._chipEls.forEach(function(el) {
        el.classList.remove("tv-active");
        el.classList.remove("tv-hidden");
      });
      this._container.classList.remove("tv-zoomed-in");
      this._hideBreadcrumb();
    }
    return new Promise(function(resolve) {
      setTimeout(resolve, 300);
    });
  };
  CardViz.prototype._showBreadcrumb = function(text) {
    if (!this._breadcrumb) {
      this._breadcrumb = document.createElement("div");
      this._breadcrumb.classList.add("tv-breadcrumb");
      this._container.insertBefore(this._breadcrumb, this._container.children[0]);
    }
    this._breadcrumb.textContent = text;
    this._breadcrumb.style.display = "block";
  };
  CardViz.prototype._hideBreadcrumb = function() {
    if (this._breadcrumb) this._breadcrumb.style.display = "none";
  };
  CardViz.prototype.destroy = function() {
    this._destroyed = true;
    this._chips.forEach(function(chip) {
      chip.reset();
    });
    this._chips = [];
    this._chipEls = [];
    this._breadcrumb = null;
    this._container.replaceChildren();
  };

  // src/system.js
  function SystemViz(container, config) {
    this._container = container;
    this._topo = loadTopology(config);
    this._cards = [];
    this._cardEls = [];
    this._breadcrumb = null;
    this._destroyed = false;
    this._init();
  }
  SystemViz.prototype._init = function() {
    const self = this;
    const topo = this._topo;
    const container = this._container;
    container.classList.add("tv-system");
    if (!Array.isArray(topo.cards)) {
      throw new Error('SystemViz requires a system-level topology with a "cards" array (e.g. "qb2", "t3000")');
    }
    topo.cards.forEach(function(cardName, i) {
      const wrapper = document.createElement("div");
      wrapper.classList.add("tv-card-wrapper");
      wrapper.dataset.cardIndex = String(i);
      const label = document.createElement("div");
      label.classList.add("tv-system-card-label");
      label.textContent = topo.labels && topo.labels[i] || "Card " + i;
      wrapper.appendChild(label);
      container.appendChild(wrapper);
      self._cardEls.push(wrapper);
      const cardContainer = document.createElement("div");
      wrapper.appendChild(cardContainer);
      const card = new CardViz(cardContainer, cardName);
      self._cards.push(card);
      if (i < topo.cards.length - 1 && topo.inter_links && topo.inter_links.length) {
        const link = document.createElement("div");
        link.classList.add("tv-system-link");
        const linkDef = topo.inter_links[0];
        link.textContent = linkDef.label || (linkDef.connector || linkDef.topology || "link");
        container.appendChild(link);
      }
    });
  };
  SystemViz.prototype.activate = function(mode) {
    if (this._destroyed) return;
    this._cards.forEach(function(card, i) {
      setTimeout(function() {
        card.activate(mode);
      }, i * 150);
    });
  };
  SystemViz.prototype.reset = function() {
    if (this._destroyed) return;
    this._cards.forEach(function(card) {
      card.reset();
    });
  };
  SystemViz.prototype.highlight = function(indices) {
    if (this._destroyed) return;
    this._cardEls.forEach(function(el, i) {
      if (indices.indexOf(i) !== -1) el.classList.add("tv-highlighted");
      else el.classList.remove("tv-highlighted");
    });
  };
  SystemViz.prototype.transitionTo = function(level, opts) {
    if (this._destroyed) return Promise.resolve();
    opts = opts || {};
    if (level === "card") {
      const idx = opts.index != null ? opts.index : 0;
      this._cardEls.forEach(function(el, i) {
        if (i === idx) {
          el.classList.add("tv-active");
          el.classList.remove("tv-hidden");
        } else {
          el.classList.remove("tv-active");
          el.classList.add("tv-hidden");
        }
      });
      this._container.classList.add("tv-zoomed-in");
      this._showBreadcrumb("System \u203A Card " + idx);
      return new Promise(function(resolve) {
        setTimeout(resolve, 300);
      });
    }
    if (level === "chip") {
      const cardIdx = opts.card != null ? opts.card : 0;
      const chipIdx = opts.chip != null ? opts.chip : 0;
      return this.transitionTo("card", { index: cardIdx }).then(function() {
        if (this._destroyed) return;
        return this._cards[cardIdx].transitionTo("chip", { index: chipIdx });
      }.bind(this));
    }
    this._cardEls.forEach(function(el) {
      el.classList.remove("tv-active");
      el.classList.remove("tv-hidden");
    });
    this._container.classList.remove("tv-zoomed-in");
    this._hideBreadcrumb();
    const resets = this._cards.map(function(card) {
      return card.transitionTo("system");
    });
    return Promise.all(resets);
  };
  SystemViz.prototype._showBreadcrumb = function(text) {
    if (!this._breadcrumb) {
      this._breadcrumb = document.createElement("div");
      this._breadcrumb.classList.add("tv-breadcrumb");
      if (this._container.children[0]) {
        this._container.insertBefore(this._breadcrumb, this._container.children[0]);
      } else {
        this._container.appendChild(this._breadcrumb);
      }
    }
    this._breadcrumb.textContent = text;
    this._breadcrumb.style.display = "block";
  };
  SystemViz.prototype._hideBreadcrumb = function() {
    if (this._breadcrumb) this._breadcrumb.style.display = "none";
  };
  SystemViz.prototype.destroy = function() {
    this._destroyed = true;
    this._cards.forEach(function(card) {
      card.destroy();
    });
    this._cards = [];
    this._cardEls = [];
    this._breadcrumb = null;
    this._container.replaceChildren();
  };

  // src/cluster.js
  var DOT_MODE_THRESHOLD = 64;
  function ClusterViz(container, config) {
    this._container = container;
    this._topo = loadTopology(config);
    this._tiles = [];
    this._activeMode = "idle";
    this._animFrame = null;
    this._breadcrumb = null;
    this._destroyed = false;
    this._zoomed = -1;
    if (this._topo.total_chips) {
      this.chipCount = this._topo.total_chips;
    } else if (typeof this._topo.chip_count === "number") {
      this.chipCount = this._topo.chip_count;
    } else if (Array.isArray(this._topo.chips)) {
      this.chipCount = this._topo.chips.length;
    } else {
      this.chipCount = (this._topo.servers || 1) * 32;
    }
    this._dotMode = this.chipCount > DOT_MODE_THRESHOLD;
    this._init();
  }
  ClusterViz.prototype._init = function() {
    const self = this;
    const container = this._container;
    const topo = this._topo;
    container.classList.add("tv-cluster");
    if (this._dotMode) container.classList.add("tv-cluster-dot-mode");
    const spec = document.createElement("div");
    spec.classList.add("tv-cluster-spec");
    spec.textContent = [
      this.chipCount + " chips",
      topo.eth_ports ? topo.eth_ports + "\xD7 " + (topo.eth_speed || "") + " ETH" : "",
      topo.topology ? topo.topology : topo.mesh_links ? "2D mesh" : ""
    ].filter(Boolean).join(" \xB7 ");
    container.appendChild(spec);
    this._cols = topo.grid ? topo.grid[1] : this._dotMode ? 16 : 8;
    this._rows = topo.grid ? topo.grid[0] : 4;
    const grid = document.createElement("div");
    grid.classList.add("tv-cluster-grid");
    grid.style.gridTemplateColumns = "repeat(" + this._cols + ", 1fr)";
    container.appendChild(grid);
    this._grid = grid;
    for (let i = 0; i < this.chipCount; i++) {
      const tile = document.createElement("div");
      tile.classList.add("tv-cluster-tile");
      tile.dataset.chipIndex = String(i);
      tile.title = "Chip " + i;
      tile.addEventListener("click", function() {
        self.transitionTo("server", { index: Math.floor(i / 32) });
      });
      grid.appendChild(tile);
      self._tiles.push(tile);
    }
    this._startAnimation();
  };
  ClusterViz.prototype._startAnimation = function() {
    const self = this;
    const tiles = this._tiles;
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
    const mode = this._activeMode;
    const cols = this._cols;
    const rows = this._rows;
    let t = 0;
    function tick() {
      if (self._destroyed) return;
      t += 0.02;
      tiles.forEach(function(tile, i) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        let heat;
        switch (mode) {
          // Inference: a wavefront sweeps left-to-right across columns
          case "inference":
            heat = Math.max(0, 1 - Math.abs(col - t % 1 * cols) / 3);
            break;
          // Diffusion: concentric rings expand outward from the grid center
          case "diffusion": {
            const cx = cols / 2, cy = rows / 2;
            const dist = Math.sqrt((col - cx) * (col - cx) + (row - cy) * (row - cy));
            const ring = t % 1 * Math.sqrt(cx * cx + cy * cy);
            heat = Math.max(0, 1 - Math.abs(dist - ring) / 2);
            break;
          }
          // Agents: sporadic random spikes that decay exponentially each frame
          case "agents":
            heat = Math.random() < 0.04 ? Math.random() : parseFloat(tile.dataset.heat || "0") * 0.85;
            break;
          // Explore: sinusoidal wave across columns, fast oscillation
          case "explore":
            heat = (Math.sin(col * 0.6 + t * Math.PI * 4) + 1) / 2;
            break;
          // Idle: very slow random flicker, low intensity, exponential decay
          default:
            heat = Math.random() < 0.02 ? Math.random() * 0.35 : parseFloat(tile.dataset.heat || "0") * 0.92;
        }
        tile.dataset.heat = String(heat.toFixed(3));
        let r, g, b;
        if (heat < 0.5) {
          const s = heat * 2;
          r = Math.round(30 + (244 - 30) * s);
          g = Math.round(74 + (196 - 74) * s);
          b = Math.round(88 + (113 - 88) * s);
        } else {
          const s = (heat - 0.5) * 2;
          r = Math.round(244 + (255 - 244) * s);
          g = Math.round(196 + (107 - 196) * s);
          b = Math.round(113 + (107 - 113) * s);
        }
        tile.style.background = "rgb(" + r + "," + g + "," + b + ")";
      });
      self._animFrame = requestAnimationFrame(tick);
    }
    this._animFrame = requestAnimationFrame(tick);
  };
  ClusterViz.prototype.activate = function(mode) {
    if (this._destroyed) return;
    this._activeMode = mode;
    this._startAnimation();
  };
  ClusterViz.prototype.reset = function() {
    if (this._destroyed) return;
    this._activeMode = "idle";
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
    this._tiles.forEach(function(tile) {
      tile.style.background = "";
      tile.dataset.heat = "0";
    });
    this._startAnimation();
  };
  ClusterViz.prototype.highlight = function(indices) {
    if (this._destroyed) return;
    this._tiles.forEach(function(tile, i) {
      if (indices.indexOf(i) !== -1) tile.classList.add("tv-highlighted");
      else tile.classList.remove("tv-highlighted");
    });
  };
  ClusterViz.prototype.transitionTo = function(level, opts) {
    if (this._destroyed) return Promise.resolve();
    opts = opts || {};
    if (level === "server") {
      const serverIdx = opts.index != null ? opts.index : 0;
      const tilesPerServer = 32;
      const start = serverIdx * tilesPerServer;
      this._tiles.forEach(function(tile, i) {
        if (i >= start && i < start + tilesPerServer) tile.classList.add("tv-active");
        else tile.classList.add("tv-hidden");
      });
      this._container.classList.add("tv-zoomed-in");
      this._zoomed = serverIdx;
      this._showBreadcrumb("Cluster \u203A Server " + serverIdx);
    } else {
      this._tiles.forEach(function(tile) {
        tile.classList.remove("tv-active");
        tile.classList.remove("tv-hidden");
      });
      this._container.classList.remove("tv-zoomed-in");
      this._zoomed = -1;
      this._hideBreadcrumb();
    }
    return new Promise(function(resolve) {
      setTimeout(resolve, 300);
    });
  };
  ClusterViz.prototype._showBreadcrumb = function(text) {
    if (!this._breadcrumb) {
      this._breadcrumb = document.createElement("div");
      this._breadcrumb.classList.add("tv-breadcrumb");
      if (this._container.children[0]) {
        this._container.insertBefore(this._breadcrumb, this._container.children[0]);
      } else {
        this._container.appendChild(this._breadcrumb);
      }
    }
    this._breadcrumb.textContent = text;
    this._breadcrumb.style.display = "block";
  };
  ClusterViz.prototype._hideBreadcrumb = function() {
    if (this._breadcrumb) this._breadcrumb.style.display = "none";
  };
  ClusterViz.prototype.destroy = function() {
    this._destroyed = true;
    if (this._animFrame) {
      cancelAnimationFrame(this._animFrame);
      this._animFrame = null;
    }
    this._tiles = [];
    this._breadcrumb = null;
    this._container.replaceChildren();
  };

  // src/index.js
  function autoInit() {
    if (typeof document === "undefined") return;
    document.querySelectorAll("[data-viz]").forEach(function(el) {
      const type = el.dataset.viz;
      const config = el.dataset.config != null ? el.dataset.config : el.dataset.arch != null ? el.dataset.arch : "bh-chip";
      const mode = el.dataset.mode != null ? el.dataset.mode : "idle";
      try {
        let viz;
        switch (type) {
          case "chip":
            viz = new TensixViz(el, { arch: config });
            viz.activate(mode);
            break;
          case "card":
            viz = new CardViz(el, config);
            viz.activate(mode);
            break;
          case "system":
            viz = new SystemViz(el, config);
            viz.activate(mode);
            break;
          case "cluster":
            viz = new ClusterViz(el, config);
            viz.activate(mode);
            break;
        }
        if (viz) el._tensixViz = viz;
      } catch (err) {
        console.warn("[tensix-viz] autoInit failed for element", el, err);
      }
    });
    TensixViz.autoInit();
  }
  if (typeof window !== "undefined" && typeof window.document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", autoInit);
    } else {
      Promise.resolve().then(autoInit);
    }
  }
  return __toCommonJS(index_exports);
})();
window.TensixViz  = _TensixVizBundle.TensixViz;
window.CardViz    = _TensixVizBundle.CardViz;
window.SystemViz  = _TensixVizBundle.SystemViz;
window.ClusterViz = _TensixVizBundle.ClusterViz;
