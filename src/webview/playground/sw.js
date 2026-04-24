// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Service worker — caches the Pyodide runtime and ttlang-sim-lite source files
// so subsequent page loads skip the ~10MB download.
//
// Cache strategy:
//   - Pyodide CDN assets:   Cache-first (they are versioned by URL; never change)
//   - ttlang-sim-lite .py:  Stale-while-revalidate (update quietly in background)

const PYODIDE_VERSION = '0.26.4';
const CACHE_PYODIDE   = `pyodide-v${PYODIDE_VERSION}`;
const CACHE_SIM_LITE  = 'ttlang-sim-lite-v6';

const PYODIDE_ORIGIN  = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Files we always pre-cache for the sim-lite package (relative to /assets/).
const SIM_LITE_MODULES = [
    'greenlet_shim',  // must be first — injected before all other modules
    'torch_compat', 'typedefs', 'constants', 'errors', 'blockstate',
    'context_types', 'diagnostics', 'debug_print', 'dfbstate',
    'trace', 'ttnnsim', 'greenlet_scheduler', 'context', 'decorators',
    'corecontext', 'dfb', 'pipe', 'sharding',
    'operation', 'copyhandlers', 'copy',
    'program', 'torch_utils', 'math', 'ttlang_sim', '__init__',
];

// ── install: pre-cache sim-lite files immediately ────────────────────────────

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_SIM_LITE).then((cache) => {
            const urls = SIM_LITE_MODULES.map(n => `/assets/ttlang-sim-lite/${n}.py`);
            urls.push('/assets/playground/pyodide-worker.js');
            return cache.addAll(urls).catch(() => { /* non-fatal on first install */ });
        })
    );
    self.skipWaiting();
});

// ── activate: prune stale caches ─────────────────────────────────────────────

self.addEventListener('activate', (event) => {
    const keep = new Set([CACHE_PYODIDE, CACHE_SIM_LITE]);
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(keys.filter(k => !keep.has(k)).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

// ── fetch: route requests to appropriate cache ───────────────────────────────

self.addEventListener('fetch', (event) => {
    const { url } = event.request;

    // Pyodide CDN: cache-first (versioned URLs never change content)
    if (url.startsWith(PYODIDE_ORIGIN)) {
        event.respondWith(cacheFirst(event.request, CACHE_PYODIDE));
        return;
    }

    // ttlang-sim-lite Python files: stale-while-revalidate
    if (url.includes('/assets/ttlang-sim-lite/') || url.includes('/assets/playground/')) {
        event.respondWith(staleWhileRevalidate(event.request, CACHE_SIM_LITE));
        return;
    }
});

async function cacheFirst(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
}

async function staleWhileRevalidate(request, cacheName) {
    const cache = await caches.open(cacheName);
    const cached = await cache.match(request);
    // Kick off revalidation in the background regardless
    const fetchPromise = fetch(request).then((response) => {
        if (response.ok) cache.put(request, response.clone());
        return response;
    }).catch(() => null);
    return cached || fetchPromise;
}
