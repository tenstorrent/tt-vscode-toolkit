#!/usr/bin/env node
/**
 * build-web.js
 *
 * Generates the GitHub Pages static site from lesson markdown files.
 *
 * Usage:
 *   node scripts/build-web.js              # build to site/
 *   node scripts/build-web.js --out /tmp/site
 *
 * Output:
 *   site/index.html                        — lesson catalog home page
 *   site/lessons/<id>/index.html           — one page per lesson
 *   site/assets/lesson-theme.css           — copied from src/webview/styles/
 *   site/assets/lesson-web-vars.css        — VSCode variable fallbacks
 *   site/assets/lesson-web.css             — web-only layout additions
 *   site/assets/lesson-web.js              — web-native interactivity
 *   site/assets/img/                       — copied from assets/img/
 *   site/assets/fonts/                     — Degular + RMMono from tt-ui (if available)
 *
 * No new npm dependencies: reuses gray-matter, marked, marked-highlight,
 * and sanitize-html which are already installed as devDependencies.
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const { marked }          = require('marked');
const { markedHighlight } = require('marked-highlight');
const matter              = require('gray-matter');
const DOMPurify           = require('isomorphic-dompurify');

/* ------------------------------------------------------------------ *
 * Paths                                                               *
 * ------------------------------------------------------------------ */

const ROOT    = path.resolve(__dirname, '..');
const TT_UI   = path.join(path.dirname(ROOT), 'tt-ui', 'src', 'fonts');

// Allow --out flag to override output directory
const outIdx = process.argv.indexOf('--out');
const SITE = outIdx !== -1
  ? path.resolve(process.argv[outIdx + 1])
  : path.join(ROOT, 'site');

const REGISTRY_PATH   = path.join(ROOT, 'content', 'lesson-registry.json');
const LESSONS_DIR     = path.join(ROOT, 'content', 'lessons');
const PAGES_DIR       = path.join(ROOT, 'content', 'pages');
const STYLES_DIR      = path.join(ROOT, 'src', 'webview', 'styles');
const SCRIPTS_DIR     = path.join(ROOT, 'src', 'webview', 'scripts');
const ASSETS_IMG_DIR  = path.join(ROOT, 'assets', 'img');
const TERM_CMDS_PATH  = path.join(ROOT, 'src', 'commands', 'terminalCommands.ts');
const MERMAID_SRC     = path.join(ROOT, 'node_modules', 'mermaid', 'dist', 'mermaid.min.js');

/* ------------------------------------------------------------------ *
 * Reference pages (content/pages/)                                    *
 * ------------------------------------------------------------------ */

const PAGES = [
  { slug: 'welcome',       title: 'Welcome',              type: 'html',     file: 'welcome.html' },
  { slug: 'faq',           title: 'FAQ',                  type: 'markdown', file: 'FAQ.md' },
  { slug: 'step-zero',     title: 'Step Zero',            type: 'markdown', file: 'step-zero.md' },
  { slug: 'riscv-guide',   title: 'RISC-V Guide',         type: 'markdown', file: 'riscv-guide.md' },
  { slug: 'version-compat',title: 'Version Compatibility',type: 'markdown', file: 'version-compatibility.md' },
];

/* ------------------------------------------------------------------ *
 * Load lesson registry                                                *
 * ------------------------------------------------------------------ */

const registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
const lessons  = registry.lessons || [];

/* ------------------------------------------------------------------ *
 * Build command text map from terminalCommands.ts                     *
 *                                                                     *
 * We parse the TypeScript source with a regex instead of importing    *
 * it (avoids a ts-node dependency at build time).                     *
 *                                                                     *
 * Extracts: KEY: { ... template: `command text` ... }                 *
 * ------------------------------------------------------------------ */

function buildCommandMap() {
  const src = fs.readFileSync(TERM_CMDS_PATH, 'utf8');
  const map = {};

  // terminalCommands.ts uses two quoting styles for template values:
  //
  //   Simple single-line commands:   template: 'pip install flask',
  //   Multi-line / interpolated:     template: `cd ~/code && ...`,
  //
  // We walk line by line, tracking the current KEY name, then extract
  // the template value regardless of which quote style is used.
  let currentKey = null;
  const keyRe = /^\s{2}([A-Z_]+):\s*\{/;
  const lines = src.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Detect a new top-level KEY: { block
    const keyMatch = line.match(keyRe);
    if (keyMatch) {
      currentKey = keyMatch[1];
    }

    if (!currentKey) continue;

    // ---- single-quoted single-line:  template: 'text',
    const sqMatch = line.match(/template:\s*'([^']+)'/);
    if (sqMatch) {
      map[currentKey] = sqMatch[1];
      currentKey = null;
      continue;
    }

    // ---- double-quoted single-line:  template: "text",
    const dqMatch = line.match(/template:\s*"([^"]+)"/);
    if (dqMatch) {
      map[currentKey] = dqMatch[1];
      currentKey = null;
      continue;
    }

    // ---- backtick single-line:  template: `text`,
    const btMatch = line.match(/template:\s*`([^`]+)`/);
    if (btMatch) {
      map[currentKey] = btMatch[1];
      currentKey = null;
      continue;
    }

    // ---- backtick multi-line:  template: `first line
    //                                        ...
    //                                        last line`,
    const btOpen = line.indexOf('template: `');
    if (btOpen !== -1) {
      let text = line.slice(btOpen + 'template: `'.length);
      while (!text.includes('`') && i < lines.length - 1) {
        i++;
        text += '\n' + lines[i];
      }
      map[currentKey] = text.slice(0, text.lastIndexOf('`'));
      currentKey = null;
    }
  }
  // --- Second pass: parse extension.ts to build a direct camelCase suffix →
  //     template mapping for commands that don't follow the simple
  //     camelToUpperSnake naming convention.
  //
  //     Pattern in extension.ts:
  //       async function createApiServerDirect() {
  //         ...
  //         const command = TERMINAL_COMMANDS.CREATE_API_SERVER.template;
  //         ...
  //       }
  //       ...
  //       registerCommand('tenstorrent.createApiServerDirect', createApiServerDirect)
  //
  //     We build:
  //       funcName → TERMINAL_COMMANDS.KEY  (from function bodies)
  //       commandSuffix → funcName           (from registerCommand calls)
  //     Then join: commandSuffix → template

  const EXTENSION_PATH = path.join(ROOT, 'src', 'extension.ts');
  if (fs.existsSync(EXTENSION_PATH)) {
    const extSrc = fs.readFileSync(EXTENSION_PATH, 'utf8');

    // Map: funcName → TERMINAL_COMMANDS key (first occurrence wins)
    const funcToKey = {};
    const funcKeyRe = /(?:async\s+)?function\s+(\w+)[^{]*\{[\s\S]*?TERMINAL_COMMANDS\.([A-Z_]+)\./g;
    let fkMatch;
    while ((fkMatch = funcKeyRe.exec(extSrc)) !== null) {
      const funcName = fkMatch[1];
      const tcKey    = fkMatch[2];
      if (!funcToKey[funcName] && map[tcKey]) {
        funcToKey[funcName] = tcKey;
      }
    }

    // Map: tenstorrent.commandSuffix → funcName (from registerCommand calls)
    const regRe = /registerCommand\(['"]tenstorrent\.(\w+)['"]\s*,\s*(\w+)/g;
    let regMatch;
    while ((regMatch = regRe.exec(extSrc)) !== null) {
      const suffix   = regMatch[1];
      const funcName = regMatch[2];
      if (funcToKey[funcName]) {
        // Store as suffix → template under a special prefix so commandTextForId
        // can find it directly without the camelToUpperSnake conversion.
        map['__ext__' + suffix] = map[funcToKey[funcName]];
      }
    }
  }

  return map;
}

const COMMAND_MAP = buildCommandMap();

// Also build a camelCase → KEY lookup so we can resolve
// tenstorrent.cloneTtLocalGenerator → CLONE_TT_LOCAL_GENERATOR
// by converting the camelCase command name to UPPER_SNAKE_CASE.
function camelToUpperSnake(str) {
  return str
    .replace(/([A-Z])/g, '_$1')
    .toUpperCase()
    .replace(/^_/, '');
}

function commandTextForId(commandId) {
  // commandId is the full VSCode command, e.g. "tenstorrent.cloneTtLocalGenerator"
  // or just the suffix "cloneTtLocalGenerator"
  const suffix = commandId.startsWith('tenstorrent.')
    ? commandId.slice('tenstorrent.'.length)
    : commandId;

  // Handle showLesson — not a terminal command
  if (suffix.startsWith('showLesson')) return null;

  // 1. Direct extension.ts mapping (most accurate — handles naming mismatches)
  if (COMMAND_MAP['__ext__' + suffix]) return COMMAND_MAP['__ext__' + suffix];

  // 2. Fallback: camelCase → UPPER_SNAKE_CASE conversion
  const key = camelToUpperSnake(suffix);
  return COMMAND_MAP[key] || null;
}

/* ------------------------------------------------------------------ *
 * Marked configuration                                                *
 * ------------------------------------------------------------------ */

// Use marked-highlight for syntax highlighting via hljs if available;
// fall back gracefully if hljs isn't present.
let hljsExtension = null;
try {
  const hljs = require('highlight.js');
  hljsExtension = markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
  });
} catch (_) {
  // highlight.js not installed — code blocks will be unstyled but functional
}

if (hljsExtension) {
  marked.use(hljsExtension);
}

/* ------------------------------------------------------------------ *
 * Link / command transformation                                        *
 *                                                                     *
 * We override marked's link renderer to intercept command: URIs.      *
 *                                                                     *
 *  showLesson?["id"]  →  <a href="/lessons/id/">Label</a>             *
 *  other command:...  →  <div class="tt-web-command"> ... </div>      *
 * ------------------------------------------------------------------ */

const WEB_RENDERER = new marked.Renderer();

// Keep track of collected command blocks so we can inject them after their
// anchor text (command buttons in flowing prose become inline code+links).
WEB_RENDERER.link = function ({ href, title, tokens }) {
  // Extract plain text from the token tree without recursing into marked.
  // Handles nested em/strong/code tokens gracefully.
  function extractText(toks) {
    return (toks || []).map(t => {
      if (t.tokens && t.tokens.length) return extractText(t.tokens);
      return t.text || t.raw || '';
    }).join('');
  }
  const text = extractText(tokens);

  if (!href || !href.startsWith('command:')) {
    const escapedHref = escapeAttr(href || '');
    const titleStr = title ? ` title="${escapeAttr(title)}"` : '';
    return `<a href="${escapedHref}"${titleStr}>${escapeHtml(text)}</a>`;
  }

  const commandUrl  = href.slice('command:'.length);
  let commandId     = commandUrl;
  let parsedArgs    = null;

  if (commandUrl.includes('?')) {
    const [cmd, args] = commandUrl.split('?', 2);
    commandId = cmd;
    try {
      parsedArgs = JSON.parse(decodeURIComponent(args));
    } catch (_) {}
  }

  // showLesson → internal navigation link
  if (commandId === 'tenstorrent.showLesson' && parsedArgs && parsedArgs[0]) {
    const lessonId = parsedArgs[0];
    return `<a href="/lessons/${lessonId}/" class="tt-lesson-link">${text}</a>`;
  }

  // Action command → terminal command display block
  const cmdText = commandTextForId(commandId);
  if (cmdText) {
    const safeId   = escapeAttr(commandId);
    const safeText = escapeHtml(cmdText);
    // VS Code badge links to the extension on the Marketplace
    const mktUrl   = 'https://marketplace.visualstudio.com/items?itemName=Tenstorrent.tenstorrent-toolkit';
    return `<div class="tt-web-command" data-command="${safeId}">` +
           `<div class="tt-web-command-header">` +
           `<span class="tt-web-command-label">${escapeHtml(text)}</span>` +
           `<div class="tt-web-command-actions">` +
           `<button class="tt-web-command-copy" title="Copy to clipboard">Copy</button>` +
           `<a class="tt-vsc-badge" href="${mktUrl}" target="_blank" rel="noopener"` +
           ` title="This button runs a command inside the VS Code extension.\nInstall the Tenstorrent toolkit to use it interactively.">` +
           `<span class="tt-vsc-badge-icon"></span>VS Code</a>` +
           `</div>` +
           `</div>` +
           `<pre class="tt-web-command-code">${safeText}</pre>` +
           `</div>`;
  }

  // Unknown command — show it as an inline code badge
  return `<code class="tt-unknown-command" title="${escapeAttr(commandId)}">${escapeHtml(text)}</code>`;
};

// Mermaid fences → preserve as raw <pre class="mermaid"> for client-side rendering
WEB_RENDERER.code = function ({ text, lang }) {
  if (lang === 'mermaid') {
    return `<pre class="mermaid">${escapeHtml(text)}</pre>\n`;
  }
  // Default: delegate to marked's default code renderer by returning false
  return false;
};

marked.use({ renderer: WEB_RENDERER });

/* ------------------------------------------------------------------ *
 * HTML utilities                                                       *
 * ------------------------------------------------------------------ */

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  return String(str || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/* ------------------------------------------------------------------ *
 * Recursive directory copy                                            *
 * ------------------------------------------------------------------ */

function copyDirRecursive(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  fs.readdirSync(src).forEach(entry => {
    const srcPath  = path.join(src, entry);
    const destPath = path.join(dest, entry);
    const stat = fs.statSync(srcPath);
    if (stat.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  });
}

/* ------------------------------------------------------------------ *
 * Render a single markdown lesson to HTML body                        *
 * ------------------------------------------------------------------ */

function renderLesson(markdownPath) {
  const raw     = fs.readFileSync(markdownPath, 'utf8');
  const { content } = matter(raw); // strip YAML frontmatter

  // marked may return a Promise in newer versions; we always call synchronously
  // since we haven't set async: true on options.
  let html = marked.parse(content);

  // Sanitize with DOMPurify, preserving our custom elements and attributes.
  // Mermaid blocks must be extracted first because their content contains
  // characters that DOMPurify would otherwise escape or strip.
  const mermaidBlocks = [];
  const MERMAID_PH = '___MERMAID_BLOCK___';
  html = html.replace(/<pre class="mermaid">([\s\S]*?)<\/pre>/g, (_m, content) => {
    mermaidBlocks.push(content);
    return `<pre class="mermaid">${MERMAID_PH}${mermaidBlocks.length - 1}</pre>`;
  });

  html = DOMPurify.sanitize(html, {
    ADD_TAGS: ['button', 'div', 'pre', 'span', 'details', 'summary'],
    ADD_ATTR: ['data-command', 'class', 'data-args', 'data-hw'],
  });

  // Restore mermaid content
  mermaidBlocks.forEach((content, i) => {
    html = html.replace(`${MERMAID_PH}${i}`, content);
  });

  return html;
}

/* ------------------------------------------------------------------ *
 * Sidebar HTML (shared across all pages)                              *
 * ------------------------------------------------------------------ */

const CATEGORY_LABELS = {
  'first-inference': 'Your First Inference',
  'serving':         'Serving & APIs',
  'compilers':       'Compilers & Frameworks',
  'cookbook':        'Cookbook',
  'applications':    'Applications',
  'advanced':        'Advanced',
  'cs-fundamentals': 'CS Fundamentals',
  'custom-training': 'Custom Training',
  'deployment':      'Deployment',
};

function categoryLabel(cat) {
  return CATEGORY_LABELS[cat] || cat.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Build the navigation sidebar HTML.
 *
 * @param {string|null} activeLessonId  - ID of the currently-active lesson, or null.
 * @param {string|null} activePageSlug  - Slug of the currently-active reference page, or null.
 */
function buildSidebar(activeLessonId, activePageSlug = null) {
  // Group lessons by category preserving registry order
  const categories = [];
  const seen = new Set();
  lessons.forEach(lesson => {
    if (!seen.has(lesson.category)) {
      seen.add(lesson.category);
      categories.push(lesson.category);
    }
  });

  let html = `<nav class="tt-sidebar" id="tt-sidebar" aria-label="Lessons">\n`;
  html += `<div class="sidebar-header">\n`;
  html += `<a href="/" class="sidebar-logo" aria-label="Tenstorrent Lessons home">`;
  html += `<span class="sidebar-logo-text">Tenstorrent<br><strong>Lessons</strong></span>`;
  html += `</a>\n`;
  html += `</div>\n`;

  categories.forEach(cat => {
    const catLessons = lessons.filter(l => l.category === cat);
    html += `<section class="sidebar-category">\n`;
    html += `<h3 class="sidebar-category-title">${escapeHtml(categoryLabel(cat))}</h3>\n`;
    html += `<ul class="sidebar-lesson-list">\n`;
    catLessons.forEach(lesson => {
      const isActive = lesson.id === activeLessonId;
      const activeClass = isActive ? ' class="active"' : '';
      const href = isActive ? '#' : `/lessons/${lesson.id}/`;
      html += `<li${activeClass}>`;
      html += `<a href="${escapeAttr(href)}"`;
      if (isActive) html += ` aria-current="page"`;
      html += `>${escapeHtml(lesson.title)}</a>`;
      html += `</li>\n`;
    });
    html += `</ul>\n</section>\n`;
  });

  // Reference pages section (welcome, FAQ, guides)
  html += `<section class="sidebar-category">\n`;
  html += `<h3 class="sidebar-category-title">Reference</h3>\n`;
  html += `<ul class="sidebar-lesson-list">\n`;
  PAGES.forEach(page => {
    const isActive = page.slug === activePageSlug;
    const activeClass = isActive ? ' class="active"' : '';
    const href = isActive ? '#' : `/${page.slug}/`;
    html += `<li${activeClass}>`;
    html += `<a href="${escapeAttr(href)}"`;
    if (isActive) html += ` aria-current="page"`;
    html += `>${escapeHtml(page.title)}</a>`;
    html += `</li>\n`;
  });
  html += `</ul>\n</section>\n`;

  html += `</nav>\n`;
  return html;
}

/* ------------------------------------------------------------------ *
 * Hardware badge HTML                                                  *
 * ------------------------------------------------------------------ */

const HW_LABELS = {
  n150:   'N150',
  n300:   'N300',
  t3k:    'T3K',
  p100:   'P100',
  p150:   'P150',
  p300:   'P300',
  p300c:  'P300C',
  p300x2: 'P300×2',
  galaxy: 'Galaxy',
};

function hwBadge(hw) {
  const label = HW_LABELS[hw] || hw.toUpperCase();
  return `<span class="hardware-chip">${escapeHtml(label)}</span>`;
}

function statusBadge(status) {
  const cls = `status-${status || 'draft'}`;
  const label = (status || 'draft').charAt(0).toUpperCase() + (status || 'draft').slice(1);
  return `<span class="status-badge ${escapeAttr(cls)}">${escapeHtml(label)}</span>`;
}

/* ------------------------------------------------------------------ *
 * Full page shell                                                      *
 * ------------------------------------------------------------------ */

function pageShell({ title, bodyClass = '', head = '', sidebar, meta = '', content }) {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)} — Tenstorrent Lessons</title>
  <link rel="stylesheet" href="/assets/lesson-web-vars.css">
  <link rel="stylesheet" href="/assets/lesson-theme.css">
  <link rel="stylesheet" href="/assets/lesson-web.css">
${head}
</head>
<body class="${escapeAttr(bodyClass)} tt-lesson-web">

<button id="sidebar-toggle" aria-expanded="false" aria-controls="tt-sidebar"
        aria-label="Toggle lesson navigation">☰</button>

${sidebar}

<main class="tt-main-content" id="main-content">
  ${meta ? `<div class="tt-lesson-meta">${meta}</div>\n` : ''}
  <div class="lesson-content">
${content}
  </div>
</main>

<script src="/assets/vendor/mermaid.min.js"></script>
<script src="/assets/lesson-web.js"></script>
</body>
</html>`;
}

/* ------------------------------------------------------------------ *
 * Generate individual lesson pages                                    *
 * ------------------------------------------------------------------ */

function buildLessonPage(lesson) {
  const markdownFile = path.join(ROOT, lesson.markdownFile);
  if (!fs.existsSync(markdownFile)) {
    console.warn(`  [SKIP] ${lesson.id}: markdown file not found at ${lesson.markdownFile}`);
    return;
  }

  const bodyHtml = renderLesson(markdownFile);

  // Meta bar: hardware badges, time estimate, status
  const hwBadges = (lesson.supportedHardware || []).map(hwBadge).join(' ');
  const timeStr  = lesson.estimatedMinutes ? `<span class="meta-time">${lesson.estimatedMinutes} min</span>` : '';
  const stsStr   = statusBadge(lesson.status);
  const metaHtml = `${hwBadges} ${timeStr} ${stsStr}`;

  // Prev / next navigation within the same category
  const catLessons = lessons.filter(l => l.category === lesson.category);
  const idx  = catLessons.findIndex(l => l.id === lesson.id);
  const prev = idx > 0 ? catLessons[idx - 1] : null;
  const next = idx < catLessons.length - 1 ? catLessons[idx + 1] : null;

  let navHtml = '';
  if (prev || next) {
    navHtml = `<nav class="lesson-nav" aria-label="Lesson navigation">`;
    navHtml += prev
      ? `<a class="nav-prev" href="/lessons/${escapeAttr(prev.id)}/">← ${escapeHtml(prev.title)}</a>`
      : `<span></span>`;
    navHtml += next
      ? `<a class="nav-next" href="/lessons/${escapeAttr(next.id)}/">${escapeHtml(next.title)} →</a>`
      : `<span></span>`;
    navHtml += `</nav>`;
  }

  const fullContent = bodyHtml + '\n' + navHtml;

  const sidebar = buildSidebar(lesson.id);
  const html = pageShell({
    title:     lesson.title,
    bodyClass: 'lesson-page',
    sidebar,
    meta:      metaHtml,
    content:   fullContent,
  });

  const outDir = path.join(SITE, 'lessons', lesson.id);
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, 'index.html'), html, 'utf8');
  console.log(`  [OK]   ${lesson.id}`);
}

/* ------------------------------------------------------------------ *
 * Generate catalog home page                                          *
 * ------------------------------------------------------------------ */

function buildHomePage() {
  // Collect all hardware values for filter chips
  const allHw = new Set();
  lessons.forEach(l => (l.supportedHardware || []).forEach(hw => allHw.add(hw)));
  const hwOrder = ['n150', 'n300', 't3k', 'p100', 'p150', 'p300', 'p300c', 'p300x2', 'galaxy'];
  const sortedHw = hwOrder.filter(hw => allHw.has(hw));

  // Filter chip bar
  let filterBar = `<div class="hw-filter-bar" role="toolbar" aria-label="Filter by hardware">\n`;
  filterBar += `  <button class="hw-filter-chip active" data-hw="all">All</button>\n`;
  sortedHw.forEach(hw => {
    const label = HW_LABELS[hw] || hw.toUpperCase();
    filterBar += `  <button class="hw-filter-chip" data-hw="${escapeAttr(hw)}">${escapeHtml(label)}</button>\n`;
  });
  filterBar += `</div>\n`;

  // Group lessons by category
  const categories = [];
  const seen = new Set();
  lessons.forEach(l => {
    if (!seen.has(l.category)) {
      seen.add(l.category);
      categories.push(l.category);
    }
  });

  let catalogHtml = '';
  catalogHtml += `<h1>Tenstorrent <strong>Lessons</strong></h1>\n`;
  catalogHtml += `<p class="catalog-intro">Interactive guides for Tenstorrent hardware and software. `;
  catalogHtml += `Use the hardware filter to find lessons for your system.</p>\n`;
  catalogHtml += filterBar;

  categories.forEach(cat => {
    const catLessons = lessons.filter(l => l.category === cat);
    catalogHtml += `<section class="lesson-category-section" data-category="${escapeAttr(cat)}">\n`;
    catalogHtml += `<h2>${escapeHtml(categoryLabel(cat))}</h2>\n`;
    catalogHtml += `<div class="lesson-card-grid">\n`;

    catLessons.forEach(lesson => {
      const hwAttr = (lesson.supportedHardware || []).join(' ');
      const badges = (lesson.supportedHardware || []).map(hwBadge).join('');
      const timeStr = lesson.estimatedMinutes ? `<span class="card-time">${lesson.estimatedMinutes} min</span>` : '';
      const statusStr = statusBadge(lesson.status);
      catalogHtml += `<a class="lesson-card" href="/lessons/${escapeAttr(lesson.id)}/" data-hw="${escapeAttr(hwAttr)}">\n`;
      catalogHtml += `  <div class="card-header">\n`;
      catalogHtml += `    <h3 class="card-title">${escapeHtml(lesson.title)}</h3>\n`;
      catalogHtml += `    <div class="card-badges">${badges} ${timeStr} ${statusStr}</div>\n`;
      catalogHtml += `  </div>\n`;
      if (lesson.description) {
        catalogHtml += `  <p class="card-desc">${escapeHtml(lesson.description)}</p>\n`;
      }
      catalogHtml += `</a>\n`;
    });

    catalogHtml += `</div>\n</section>\n`;
  });

  const sidebar = buildSidebar(null);
  const html = pageShell({
    title:     'Lessons',
    bodyClass: 'catalog-page',
    sidebar,
    content:   catalogHtml,
  });

  fs.writeFileSync(path.join(SITE, 'index.html'), html, 'utf8');
  console.log('  [OK]   index.html (catalog)');
}

/* ------------------------------------------------------------------ *
 * Copy static assets                                                  *
 * ------------------------------------------------------------------ */

function copyAssets() {
  const assetsOut = path.join(SITE, 'assets');
  fs.mkdirSync(assetsOut, { recursive: true });

  const cssFiles = [
    'lesson-theme.css',
    'lesson-web-vars.css',
  ];
  cssFiles.forEach(f => {
    const src = path.join(STYLES_DIR, f);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(assetsOut, f));
      console.log(`  [OK]   assets/${f}`);
    } else {
      console.warn(`  [MISS] assets/${f} not found at ${src}`);
    }
  });

  // Web JS
  const jsFile = path.join(SCRIPTS_DIR, 'lesson-web.js');
  if (fs.existsSync(jsFile)) {
    fs.copyFileSync(jsFile, path.join(assetsOut, 'lesson-web.js'));
    console.log('  [OK]   assets/lesson-web.js');
  }

  // Web-only layout CSS (generated inline below)
  fs.writeFileSync(path.join(assetsOut, 'lesson-web.css'), webLayoutCss(), 'utf8');
  console.log('  [OK]   assets/lesson-web.css');

  // Images (recursive copy to handle subdirectories)
  const imgOut = path.join(assetsOut, 'img');
  if (fs.existsSync(ASSETS_IMG_DIR)) {
    copyDirRecursive(ASSETS_IMG_DIR, imgOut);
    console.log(`  [OK]   assets/img/ (recursive)`);
  }

  // Mermaid.js (vendored from node_modules — ~2.7 MB, no CDN dependency)
  const vendorOut = path.join(assetsOut, 'vendor');
  fs.mkdirSync(vendorOut, { recursive: true });
  if (fs.existsSync(MERMAID_SRC)) {
    fs.copyFileSync(MERMAID_SRC, path.join(vendorOut, 'mermaid.min.js'));
    console.log('  [OK]   assets/vendor/mermaid.min.js');
  } else {
    console.warn('  [MISS] mermaid.min.js — run npm install first');
  }

  // Fonts from tt-ui (Degular + RMMono + IBM Plex Mono) — optional
  const WANTED_FONTS = [
    'DegularDisplay-Light.otf',
    'DegularDisplay-Medium.otf',
    'DegularDisplay-Semibold.otf',
    'DegularText-Medium.otf',
    'DegularText-Bold.otf',
    'DegularText-MediumItalic.otf',
    'RMMono-Regular.otf',
    'RMMono-SemiBold.otf',
    'IBMPlexMono-Regular.ttf',
    'IBMPlexMono-Medium.ttf',
    'IBMPlexMono-SemiBold.ttf',
    'IBMPlexMono-Bold.ttf',
    'IBMPlexMono-Italic.ttf',
  ];
  if (fs.existsSync(TT_UI)) {
    const fontsOut = path.join(assetsOut, 'fonts');
    fs.mkdirSync(fontsOut, { recursive: true });
    let copied = 0;
    WANTED_FONTS.forEach(f => {
      const src = path.join(TT_UI, f);
      if (fs.existsSync(src)) {
        fs.copyFileSync(src, path.join(fontsOut, f));
        copied++;
      }
    });
    console.log(`  [OK]   assets/fonts/ (${copied} font files from tt-ui)`);
  } else {
    console.log('  [SKIP] assets/fonts/ — tt-ui not found at', TT_UI);
  }
}

/* ------------------------------------------------------------------ *
 * Web-only layout CSS                                                  *
 *                                                                     *
 * lesson-theme.css handles content styles; this file adds the         *
 * two-column sidebar layout, mobile toggle, and web-command blocks.   *
 * ------------------------------------------------------------------ */

function webLayoutCss() {
  return `/**
 * lesson-web.css — web-only layout additions for the GitHub Pages site.
 * Content styles live in lesson-theme.css; this file adds the two-column
 * layout, sidebar, mobile toggle, catalog cards, and terminal command blocks.
 */

/* ===== Page layout ===== */

body.tt-lesson-web {
  display: grid;
  grid-template-columns: 260px 1fr;
  grid-template-rows: auto;
  grid-template-areas: "sidebar main";
  min-height: 100vh;
  padding: 0;
  gap: 0;
}

/* ===== Sidebar ===== */

.tt-sidebar {
  grid-area: sidebar;
  background: var(--vscode-sideBar-background, #111827);
  border-right: 1px solid var(--tt-border, rgba(255,255,255,0.1));
  overflow-y: auto;
  position: sticky;
  top: 0;
  height: 100vh;
  padding: 0 0 32px;
}

.sidebar-header {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--tt-border, rgba(255,255,255,0.1));
  position: sticky;
  top: 0;
  background: var(--vscode-sideBar-background, #111827);
  z-index: 10;
}

.sidebar-logo {
  color: var(--vscode-foreground);
  text-decoration: none;
  font-size: 0.85rem;
  line-height: 1.3;
  display: block;
}

.sidebar-logo-text {
  display: block;
  font-weight: 300;
  letter-spacing: 0.02em;
}

.sidebar-logo-text strong {
  font-weight: 700;
  color: var(--tt-primary);
}

.sidebar-category {
  padding: 16px 0 8px;
}

.sidebar-category-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--tt-muted, rgba(226,232,240,0.5));
  padding: 0 16px 4px;
  margin: 0;
  border-bottom: none;
}

.sidebar-lesson-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar-lesson-list li {
  margin: 0;
  padding: 0;
}

.sidebar-lesson-list li a {
  display: block;
  padding: 7px 16px;
  font-size: 0.82rem;
  color: var(--vscode-foreground);
  text-decoration: none;
  border-left: 3px solid transparent;
  transition: background 0.15s, border-color 0.15s;
  line-height: 1.4;
}

.sidebar-lesson-list li a:hover {
  background: var(--tt-hover-bg);
  text-decoration: none;
}

.sidebar-lesson-list li.active a,
.sidebar-lesson-list li a[aria-current="page"] {
  border-left-color: var(--tt-primary);
  color: var(--tt-primary);
  background: var(--tt-hover-bg);
  font-weight: 600;
}

/* ===== Main content area ===== */

.tt-main-content {
  grid-area: main;
  overflow-y: auto;
  padding: 0;
}

.tt-lesson-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px 32px;
  background: var(--vscode-panel-background, #151e2b);
  border-bottom: 1px solid var(--tt-border, rgba(255,255,255,0.1));
}

.meta-time {
  font-size: 0.8rem;
  color: var(--tt-muted);
  padding: 4px 8px;
  background: var(--tt-hover-bg);
  border-radius: 4px;
}

/* ===== Mobile sidebar toggle ===== */

#sidebar-toggle {
  display: none;
  position: fixed;
  top: 12px;
  left: 12px;
  z-index: 100;
  background: var(--tt-primary);
  color: white;
  border: none;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 1.1rem;
  cursor: pointer;
  line-height: 1;
}

/* ===== Terminal command blocks ===== */

.tt-web-command {
  background: var(--tt-code-bg, #0f1923);
  border: 1px solid var(--tt-border, rgba(255,255,255,0.1));
  border-left: 4px solid var(--tt-primary);
  border-radius: 4px;
  margin: 16px 0;
  overflow: hidden;
}

.tt-web-command-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: var(--tt-hover-bg);
  border-bottom: 1px solid var(--tt-border, rgba(255,255,255,0.06));
}

.tt-web-command-label {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--tt-primary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tt-web-command-copy {
  background: var(--tt-primary);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}

.tt-web-command-copy:hover {
  background: var(--tt-primary-light);
}

.tt-web-command-copy.copied {
  background: var(--tt-success);
}

.tt-web-command-code {
  margin: 0;
  padding: 14px;
  background: transparent;
  border: none;
  border-radius: 0;
  font-size: 0.88em;
  white-space: pre-wrap;
  word-break: break-all;
}

/* ===== Code block wrapper (for injected copy buttons) ===== */

.code-block-wrapper {
  position: relative;
}

.code-block-wrapper .copy-button {
  position: absolute;
  top: 8px;
  right: 8px;
  background: var(--tt-primary);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s, background 0.15s;
}

.code-block-wrapper:hover .copy-button,
.code-block-wrapper .copy-button:focus {
  opacity: 1;
}

.code-block-wrapper .copy-button.copied {
  background: var(--tt-success);
}

/* ===== Catalog / home page ===== */

.catalog-intro {
  color: var(--tt-muted);
  margin-bottom: 24px;
  font-size: 1.05rem;
}

.hw-filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 32px;
}

.hw-filter-chip {
  background: var(--tt-hover-bg);
  color: var(--vscode-foreground);
  border: 1px solid var(--tt-border, rgba(255,255,255,0.15));
  border-radius: 20px;
  padding: 6px 16px;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}

.hw-filter-chip:hover {
  background: color-mix(in srgb, var(--tt-primary) 15%, var(--vscode-background));
  border-color: var(--tt-primary);
  color: var(--tt-primary);
}

.hw-filter-chip.active {
  background: var(--tt-primary);
  border-color: var(--tt-primary);
  color: white;
}

.lesson-category-section {
  margin-bottom: 48px;
}

.lesson-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.lesson-card {
  display: block;
  background: var(--vscode-editor-background);
  border: 1px solid var(--tt-border, rgba(255,255,255,0.1));
  border-radius: 8px;
  padding: 16px;
  text-decoration: none;
  color: var(--vscode-foreground);
  transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
}

.lesson-card:hover {
  border-color: var(--tt-primary);
  box-shadow: 0 4px 16px rgba(50, 147, 178, 0.15);
  transform: translateY(-2px);
  text-decoration: none;
}

.card-header {
  margin-bottom: 8px;
}

.card-title {
  font-size: 0.95rem;
  font-weight: 600;
  margin: 0 0 8px;
  border: none;
  padding: 0;
  color: var(--vscode-foreground);
  line-height: 1.3;
}

.card-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.card-time {
  font-size: 0.72rem;
  color: var(--tt-muted);
}

.card-desc {
  font-size: 0.82rem;
  color: var(--tt-muted);
  margin: 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ===== Responsive ===== */

@media (max-width: 768px) {
  body.tt-lesson-web {
    grid-template-columns: 1fr;
    grid-template-areas:
      "main";
  }

  .tt-sidebar {
    position: fixed;
    top: 0;
    left: -280px;
    width: 260px;
    height: 100vh;
    z-index: 50;
    transition: left 0.25s ease;
    box-shadow: 2px 0 16px rgba(0, 0, 0, 0.4);
  }

  .tt-sidebar.sidebar-open {
    left: 0;
  }

  #sidebar-toggle {
    display: block;
  }

  .tt-main-content {
    padding-top: 52px; /* clear the toggle button */
  }

  .tt-lesson-meta {
    padding: 10px 16px;
  }

  .lesson-content {
    padding: 16px;
  }

  .lesson-card-grid {
    grid-template-columns: 1fr;
  }
}

/* ===== VS Code badge (on command blocks) ===== */

/* The actions row houses the copy button + VS Code badge side by side. */
.tt-web-command-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

/* The VS Code badge — links to the Marketplace, signals the button is
   fully interactive inside the VS Code extension. */
.tt-vsc-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #007acc;               /* VS Code blue */
  background: rgba(0, 122, 204, 0.12);
  border: 1px solid rgba(0, 122, 204, 0.35);
  border-radius: 4px;
  padding: 3px 8px;
  text-decoration: none;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
}

.tt-vsc-badge:hover {
  background: rgba(0, 122, 204, 0.22);
  border-color: rgba(0, 122, 204, 0.65);
  color: #007acc;
  text-decoration: none;
}

/* Small VS Code "⧉" icon placeholder — pure CSS, no image asset needed */
.tt-vsc-badge-icon::before {
  content: "⧉";
  font-size: 0.9em;
}

/* ===== VSCode-only elements (disabled on web, tooltip on hover) ===== */

/* Elements with data-vscode-command are interactive only inside the
   VS Code extension.  On the web they're shown in a muted, disabled
   state with an explanatory tooltip on hover. */
[data-vscode-command] {
  opacity: 0.5;
  cursor: not-allowed !important;
  pointer-events: auto;          /* must stay 'auto' so ::after tooltip works */
  position: relative;
  filter: grayscale(0.3);
}

/* The tooltip that appears on hover */
[data-vscode-command]:hover::after {
  content: "Available in the VS Code extension";
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--vscode-editor-background, #1a2332);
  color: var(--vscode-foreground, #e2e8f0);
  border: 1px solid var(--tt-primary, #3293b2);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
  pointer-events: none;
  z-index: 200;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

/* Small VS Code branding on the tooltip */
[data-vscode-command]:hover::before {
  content: "⧉ VS Code";
  position: absolute;
  bottom: calc(100% + 32px);    /* sits above the main tooltip */
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 122, 204, 0.15);
  border: 1px solid rgba(0, 122, 204, 0.4);
  color: #007acc;
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  white-space: nowrap;
  pointer-events: none;
  z-index: 201;
}

/* ===== Mermaid diagram overrides ===== */

/* Mermaid renders SVG inline; give it a bit of breathing room and
   constrain its max-width so it doesn't overflow on narrow viewports. */
.mermaid {
  max-width: 100%;
  overflow-x: auto;
  margin: 16px 0;
  padding: 0;
  background: transparent;
  border: none;
}

.mermaid svg {
  max-width: 100%;
  height: auto;
}

/* ===== Reference page overrides ===== */

/* welcome.html has its own inline <style> that sets body { max-width: 900px }.
   On the reference page, the .lesson-content wrapper provides the constraint
   so we need to undo the body-level max-width. */
.reference-page .lesson-content {
  max-width: 860px;
}

/* walkthrough-item list items in welcome.html get a pointer cursor since
   their onclick now navigates.  Undo the cursor:not-allowed from disabled
   rule in case they also happen to have data-vscode-command (they don't,
   but defensive CSS). */
.walkthrough-item {
  transition: background 0.2s, border-color 0.2s, transform 0.15s;
}
`;

}

/* ------------------------------------------------------------------ *
 * Welcome.html transformation                                         *
 *                                                                     *
 * welcome.html is a self-contained VSCode webview page with inline    *
 * <style>, <script>, and VSCode API calls. We extract the body        *
 * content and rewrite the interactive parts for the web:              *
 *                                                                     *
 *  openWalkthrough('id')      → window.location='/lessons/id/'        *
 *  executeCommand('show*')    → window.location='/page-slug/'         *
 *  executeCommand('<other>')  → data-vscode-command attribute          *
 *                              (disabled via CSS, badge on hover)     *
 *  <script> block             → removed (VSCode API not available)    *
 * ------------------------------------------------------------------ */

/** VSCode commands that correspond to reference pages on this site. */
const COMMAND_PAGE_MAP = {
  'tenstorrent.showStepZero': '/step-zero/',
  'tenstorrent.showFaq':      '/faq/',
  'tenstorrent.showWelcome':  '/welcome/',
};

function transformWelcomeHtml(rawHtml) {
  // 1. Extract the <style> block from <head> so we can include it inline.
  //    The custom CSS inside welcome.html uses VSCode CSS variables that our
  //    lesson-web-vars.css already provides fallbacks for.
  let headStyle = '';
  rawHtml = rawHtml.replace(/<head[\s\S]*?<\/head>/i, head => {
    const m = head.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
    if (m) headStyle = `<style>\n${m[1]}\n</style>\n`;
    return '';
  });

  // 2. Extract <body> content
  const bodyMatch = rawHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  let body = bodyMatch ? bodyMatch[1] : rawHtml;

  // 3. Remove the VSCode API <script> block (acquireVsCodeApi, openWalkthrough,
  //    executeCommand functions — they don't exist on the web).
  body = body.replace(/<script[\s\S]*?<\/script>/gi, '');

  // 4. Transform openWalkthrough('lessonId') → navigate to lesson page.
  //    These onclick attrs appear on <li class="walkthrough-item"> elements.
  body = body.replace(
    /onclick="openWalkthrough\('([^']+)'\)"/g,
    (_, lessonId) => `onclick="window.location='/lessons/${lessonId}/'" style="cursor:pointer"`
  );

  // 5. Transform executeCommand('cmd') onclick handlers.
  //    Known page-mapping commands → navigate; others → disabled (data attr).
  //    Pattern covers both: onclick="executeCommand('cmd')"
  //    and the semicolon variant: onclick="executeCommand('cmd'); return false;"
  body = body.replace(
    /onclick="executeCommand\('([^']+)'\)(?:;\s*return false;)?"/g,
    (_, cmdId) => {
      const page = COMMAND_PAGE_MAP[cmdId];
      if (page) {
        return `onclick="window.location='${page}'"`;
      }
      // Mark as VSCode-only — CSS will style these with a disabled appearance
      // and a tooltip. The data attribute preserves the original command for
      // potential future use (e.g. deep-linking into the extension).
      return `data-vscode-command="${escapeAttr(cmdId)}" onclick="return false;"`;
    }
  );

  // 6. Same for href="#" onclick="executeCommand('cmd')..." patterns on <a> tags.
  body = body.replace(
    /href="#"\s+onclick="executeCommand\('([^']+)'\)[^"]*"/g,
    (_, cmdId) => {
      const page = COMMAND_PAGE_MAP[cmdId];
      if (page) return `href="${page}"`;
      return `href="#" onclick="return false;" data-vscode-command="${escapeAttr(cmdId)}"`;
    }
  );

  // Alternate order: onclick="..." href="#"
  body = body.replace(
    /onclick="executeCommand\('([^']+)'\)[^"]*"\s+href="#"/g,
    (_, cmdId) => {
      const page = COMMAND_PAGE_MAP[cmdId];
      if (page) return `href="${page}"`;
      return `href="#" onclick="return false;" data-vscode-command="${escapeAttr(cmdId)}"`;
    }
  );

  return headStyle + body;
}

/* ------------------------------------------------------------------ *
 * Render a reference markdown page (FAQ, step-zero, etc.)            *
 * ------------------------------------------------------------------ */

function renderMarkdownPage(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const { content } = matter(raw);
  let html = marked.parse(content);

  // Same mermaid-safe DOMPurify pass as renderLesson
  const mermaidBlocks = [];
  const MERMAID_PH = '___MERMAID_BLOCK___';
  html = html.replace(/<pre class="mermaid">([\s\S]*?)<\/pre>/g, (_m, mc) => {
    mermaidBlocks.push(mc);
    return `<pre class="mermaid">${MERMAID_PH}${mermaidBlocks.length - 1}</pre>`;
  });
  html = DOMPurify.sanitize(html, {
    ADD_TAGS: ['button', 'div', 'pre', 'span', 'details', 'summary'],
    ADD_ATTR: ['data-command', 'class', 'data-args', 'data-hw'],
  });
  mermaidBlocks.forEach((mc, i) => {
    html = html.replace(`${MERMAID_PH}${i}`, mc);
  });

  return html;
}

/* ------------------------------------------------------------------ *
 * Generate reference pages (content/pages/)                          *
 * ------------------------------------------------------------------ */

function buildPages() {
  PAGES.forEach(page => {
    const filePath = path.join(PAGES_DIR, page.file);
    if (!fs.existsSync(filePath)) {
      console.warn(`  [SKIP] page ${page.slug}: file not found at ${filePath}`);
      return;
    }

    let bodyContent;
    let extraHead = '';

    if (page.type === 'html') {
      const raw = fs.readFileSync(filePath, 'utf8');
      bodyContent = transformWelcomeHtml(raw);
      // transformWelcomeHtml prepends any <style> block inline in bodyContent
    } else {
      bodyContent = renderMarkdownPage(filePath);
    }

    const sidebar = buildSidebar(null, page.slug);
    const html = pageShell({
      title:     page.title,
      bodyClass: `reference-page page-${page.slug}`,
      sidebar,
      head:      extraHead,
      content:   bodyContent,
    });

    const outDir = path.join(SITE, page.slug);
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'index.html'), html, 'utf8');
    console.log(`  [OK]   ${page.slug}/index.html`);
  });
}

/* ------------------------------------------------------------------ *
 * Main build                                                          *
 * ------------------------------------------------------------------ */

function build() {
  console.log(`\nBuilding Tenstorrent Lessons site → ${SITE}\n`);

  fs.mkdirSync(SITE, { recursive: true });

  console.log('Assets:');
  copyAssets();

  console.log('\nLessons:');
  lessons.forEach(buildLessonPage);

  console.log('\nCatalog:');
  buildHomePage();

  console.log('\nReference pages:');
  buildPages();

  // Emit a simple 404 page
  const notFoundHtml = pageShell({
    title: '404 Not Found',
    bodyClass: 'error-page',
    sidebar: buildSidebar(null),
    content: `<h1>404 <strong>Not Found</strong></h1>
<p>The page you're looking for doesn't exist.</p>
<p><a href="/">Back to lessons</a></p>`,
  });
  fs.writeFileSync(path.join(SITE, '404.html'), notFoundHtml, 'utf8');
  console.log('\n  [OK]   404.html');

  // Emit a .nojekyll file so GitHub Pages serves _ prefixed files correctly
  fs.writeFileSync(path.join(SITE, '.nojekyll'), '', 'utf8');
  console.log('  [OK]   .nojekyll');

  console.log(`\nDone. ${lessons.length} lessons + ${PAGES.length} reference pages built.\n`);
}

build();
