/**
 * lesson-web.js
 *
 * Web-native interactivity for the GitHub Pages lesson site.
 * Replaces lesson-viewer.js (which uses acquireVsCodeApi() and cannot run
 * outside the VSCode webview context).
 *
 * Responsibilities:
 *  - Copy-to-clipboard for terminal command blocks (.tt-web-command-copy)
 *  - Copy-to-clipboard for code block copy buttons (.copy-button)
 *  - Hardware filter chip toggling on the catalog home page
 *  - Mobile sidebar toggle
 *  - Smooth scroll-to-heading for anchor links
 *  - Mermaid diagram initialisation (loaded separately via CDN)
 */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ *
   * Clipboard helper                                                     *
   * ------------------------------------------------------------------ */

  /**
   * Copy text to the clipboard, falling back to execCommand for older browsers.
   * Returns a Promise<void> that resolves when the copy succeeds.
   */
  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for HTTP contexts or older browsers
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(textarea);
      ok ? resolve() : reject(new Error('execCommand copy failed'));
    });
  }

  /**
   * Briefly show a "Copied!" confirmation on a button, then restore its label.
   */
  function flashCopied(btn) {
    var original = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(function () {
      btn.textContent = original;
      btn.classList.remove('copied');
    }, 1800);
  }

  /* ------------------------------------------------------------------ *
   * Terminal command blocks                                              *
   * ------------------------------------------------------------------ */

  /**
   * Wire up copy buttons inside .tt-web-command blocks.
   * Each block contains a <pre class="tt-web-command-code"> with the shell
   * command text, and a <button class="tt-web-command-copy"> to copy it.
   */
  function initCommandBlocks() {
    document.querySelectorAll('.tt-web-command-copy').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var block = btn.closest('.tt-web-command');
        if (!block) return;
        var pre = block.querySelector('.tt-web-command-code');
        if (!pre) return;
        copyToClipboard(pre.textContent.trim()).then(function () {
          flashCopied(btn);
        }).catch(function () {
          btn.textContent = 'Failed';
          setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
        });
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Code block copy buttons                                              *
   * ------------------------------------------------------------------ */

  /**
   * Add a copy button to every <pre><code> block that doesn't already have one.
   * The build script may pre-generate these; this function is a fallback for
   * any that were missed.
   */
  function initCodeBlockCopyButtons() {
    document.querySelectorAll('pre').forEach(function (pre) {
      // Skip command blocks (they have their own copy button)
      if (pre.classList.contains('tt-web-command-code')) return;
      // Skip if a copy button already exists
      if (pre.querySelector('.copy-button')) return;

      var btn = document.createElement('button');
      btn.className = 'copy-button';
      btn.textContent = 'Copy';
      btn.setAttribute('aria-label', 'Copy code to clipboard');

      btn.addEventListener('click', function () {
        var code = pre.querySelector('code');
        var text = (code || pre).textContent;
        copyToClipboard(text).then(function () {
          flashCopied(btn);
        }).catch(function () {
          btn.textContent = 'Failed';
          setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
        });
      });

      // Wrap pre in a relative container so the button can be positioned
      var wrapper = pre.parentNode;
      if (!wrapper.classList.contains('code-block-wrapper')) {
        var div = document.createElement('div');
        div.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(div, pre);
        div.appendChild(pre);
        wrapper = div;
      }
      wrapper.appendChild(btn);
    });
  }

  /* ------------------------------------------------------------------ *
   * Hardware filter chips (catalog / home page)                          *
   * ------------------------------------------------------------------ */

  /**
   * Activate hardware filter chip behaviour on the home page.
   *
   * Chips have data-hw="<hardware-id>" (or data-hw="all").
   * Lesson cards have data-hw="<space-separated list of hardware ids>".
   *
   * Clicking a chip hides cards that do not support that hardware.
   * "All" shows everything.
   */
  function initHardwareFilter() {
    var chips = document.querySelectorAll('.hw-filter-chip');
    if (!chips.length) return;

    var cards = document.querySelectorAll('.lesson-card');

    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        var hw = chip.getAttribute('data-hw');

        // Update active chip
        chips.forEach(function (c) { c.classList.remove('active'); });
        chip.classList.add('active');

        // Filter cards
        cards.forEach(function (card) {
          if (hw === 'all') {
            card.style.display = '';
          } else {
            var supported = (card.getAttribute('data-hw') || '').split(' ');
            card.style.display = supported.indexOf(hw) !== -1 ? '' : 'none';
          }
        });

        // Show/hide empty category sections
        document.querySelectorAll('.lesson-category-section').forEach(function (section) {
          // A card with display:none from the filter still exists in the DOM;
          // check inline style directly since querySelectorAll :visible is non-standard.
          var anyVisible = false;
          section.querySelectorAll('.lesson-card').forEach(function (c) {
            if (c.style.display !== 'none') anyVisible = true;
          });
          section.style.display = anyVisible ? '' : 'none';
        });
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Sidebar: mobile toggle                                               *
   * ------------------------------------------------------------------ */

  function initSidebarToggle() {
    var toggle = document.getElementById('sidebar-toggle');
    var sidebar = document.getElementById('tt-sidebar');
    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', function () {
      var open = sidebar.classList.toggle('sidebar-open');
      toggle.setAttribute('aria-expanded', String(open));
    });

    // Close sidebar when a lesson link is clicked (mobile UX)
    sidebar.querySelectorAll('a').forEach(function (a) {
      a.addEventListener('click', function () {
        sidebar.classList.remove('sidebar-open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Smooth anchor scrolling                                              *
   * ------------------------------------------------------------------ */

  function initAnchorScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (a) {
      a.addEventListener('click', function (e) {
        var id = a.getAttribute('href').slice(1);
        var target = document.getElementById(id);
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Update URL without adding a history entry
        history.replaceState(null, '', '#' + id);
      });
    });
  }

  /* ------------------------------------------------------------------ *
   * Mermaid initialisation                                               *
   * ------------------------------------------------------------------ */

  /**
   * If Mermaid was loaded (via CDN script tag in the page head), initialise it.
   * The build script emits <div class="mermaid"> blocks; Mermaid finds them
   * automatically if mermaid.initialize() is called with startOnLoad: true.
   * This function is a safety net in case the auto-init flag wasn't set.
   */
  function initMermaid() {
    if (typeof mermaid === 'undefined') return;
    if (document.querySelectorAll('.mermaid').length === 0) return;
    try {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        themeVariables: {
          primaryColor: '#3293b2',
          primaryTextColor: '#e2e8f0',
          primaryBorderColor: '#3fb7de',
          lineColor: '#63c7e9',
          sectionBkgColor: '#1a2332',
          altSectionBkgColor: '#151e2b',
          gridColor: '#2d3748',
          secondaryColor: '#2d3748',
          tertiaryColor: '#111827',
        },
      });
      mermaid.run({ querySelector: '.mermaid' });
    } catch (err) {
      console.warn('Mermaid init failed:', err);
    }
  }

  /* ------------------------------------------------------------------ *
   * Boot                                                                 *
   * ------------------------------------------------------------------ */

  function init() {
    initCommandBlocks();
    initCodeBlockCopyButtons();
    initHardwareFilter();
    initSidebarToggle();
    initAnchorScroll();
    initMermaid();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
