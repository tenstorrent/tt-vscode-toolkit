/**
 * Tensix Viz Rendering Tests
 *
 * Verifies that `tensix_viz` fenced code blocks in lesson markdown are
 * correctly converted to canvas HTML by MarkdownRenderer and that
 * sanitize-html preserves the data-arch, data-script attributes and
 * the <canvas> element needed by tensix-viz.js autoInit().
 */

import { expect } from 'chai';
import * as path from 'path';
import { MarkdownRenderer } from '../../src/renderers/MarkdownRenderer';

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/** Minimal valid tensix_viz script payload */
const SAMPLE_SCRIPT = JSON.stringify([
  { step: 'highlight', cores: [[0, 0]], label: 'Core 0,0' },
  { step: 'pause', ms: 200 },
]);

function makeFence(lang: string, body: string): string {
  return `\`\`\`${lang}\n${body}\n\`\`\``;
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

describe('Tensix Viz Rendering', () => {
  let renderer: MarkdownRenderer;

  before(() => {
    renderer = new MarkdownRenderer({ sanitize: true, enableHighlight: true });
  });

  // ── HTML structure ──────────────────────────────────────────────────────────

  it('should produce a tensix-viz-container div for a tensix_viz fence', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('tensix-viz-container');
  });

  it('should include a <canvas> element inside the container', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.match(/<canvas[^>]*class="tensix-viz-canvas"[^>]*>/);
  });

  it('should preserve the data-arch attribute on the container', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('data-arch="wormhole"');
  });

  it('should preserve the data-script attribute on the container', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('data-script=');
  });

  // ── arch variants ───────────────────────────────────────────────────────────

  it('should use wormhole arch badge for wormhole chips', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('Wormhole');
    expect(html).not.to.include('Blackhole');
  });

  it('should use blackhole arch badge for blackhole chips', async () => {
    const md = makeFence('tensix_viz arch=blackhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('Blackhole');
  });

  it('should default to wormhole when arch is omitted', async () => {
    const md = makeFence('tensix_viz', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('data-arch="wormhole"');
  });

  // ── JSON payload handling ───────────────────────────────────────────────────

  it('should embed valid JSON script payload in the data-script attribute', async () => {
    const script = [{ step: 'clear' }];
    const md = makeFence('tensix_viz arch=wormhole', JSON.stringify(script));
    const { html } = await renderer.render(md);
    // The JSON is HTML-attribute-encoded; confirm the step key is present
    expect(html).to.include('step');
    expect(html).to.include('clear');
  });

  it('should fall back to empty array for malformed JSON in fence body', async () => {
    const md = makeFence('tensix_viz arch=wormhole', '{ this is not json }');
    const { html } = await renderer.render(md);
    // data-script should be set to the encoded empty array []
    expect(html).to.include('data-script=');
    // Container still rendered — visualizer handles empty scripts gracefully
    expect(html).to.include('tensix-viz-container');
  });

  // ── Controls ────────────────────────────────────────────────────────────────

  it('should include play and step buttons', async () => {
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    expect(html).to.include('tv-play');
    expect(html).to.include('tv-step');
  });

  // ── Non-tensix_viz fences are unaffected ────────────────────────────────────

  it('should not produce canvas elements for regular code fences', async () => {
    const md = makeFence('python', 'print("hello")');
    const { html } = await renderer.render(md);
    expect(html).not.to.include('tensix-viz-container');
    expect(html).not.to.include('<canvas');
  });

  // ── Sanitize-html allowlist ─────────────────────────────────────────────────

  it('should not strip canvas elements when sanitize is enabled', async () => {
    // Renderer is constructed with sanitize: true (default)
    const md = makeFence('tensix_viz arch=wormhole', SAMPLE_SCRIPT);
    const { html } = await renderer.render(md);
    // If canvas were stripped, the word "canvas" would not appear in html
    expect(html.toLowerCase()).to.include('canvas');
  });

  it('should preserve data-arch through sanitize-html', async () => {
    const renderer2 = new MarkdownRenderer({ sanitize: true });
    const md = makeFence('tensix_viz arch=blackhole', SAMPLE_SCRIPT);
    const { html } = await renderer2.render(md);
    expect(html).to.include('data-arch="blackhole"');
  });

  // ── Lesson file scan ────────────────────────────────────────────────────────

  describe('Lesson tensix_viz fence validation', () => {
    const fs = require('fs') as typeof import('fs');
    const lessonsDir = path.join(__dirname, '../../content/lessons');
    const lessonFiles = fs.readdirSync(lessonsDir).filter((f: string) => f.endsWith('.md'));

    lessonFiles.forEach((file: string) => {
      it(`${file} — all tensix_viz fence bodies should be valid JSON`, () => {
        const content = fs.readFileSync(path.join(lessonsDir, file), 'utf8');
        const lines = content.split('\n');
        let inVizFence = false;
        let bodyLines: string[] = [];

        for (const line of lines) {
          const trimmed = line.trim();
          if (!inVizFence && trimmed.startsWith('```tensix_viz')) {
            inVizFence = true;
            bodyLines = [];
          } else if (inVizFence && trimmed === '```') {
            const body = bodyLines.join('\n').trim();
            if (body.length > 0) {
              let parsed: unknown;
              try {
                parsed = JSON.parse(body);
              } catch (e) {
                throw new Error(
                  `${file}: tensix_viz fence body is not valid JSON:\n${body.slice(0, 200)}`
                );
              }
              expect(Array.isArray(parsed), `${file}: tensix_viz script must be a JSON array`).to.be.true;
              for (const step of parsed as unknown[]) {
                expect(typeof step === 'object' && step !== null,
                  `${file}: each tensix_viz step must be an object`).to.be.true;
                expect('step' in (step as object),
                  `${file}: each tensix_viz step must have a "step" key`).to.be.true;
              }
            }
            inVizFence = false;
          } else if (inVizFence) {
            bodyLines.push(line);
          }
        }
      });
    });
  });
});
