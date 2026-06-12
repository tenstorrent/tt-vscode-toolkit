# PR: Prose copyedit (`vscode-toolkit-copyedit` → `main`)

Use this as the GitHub PR description.

## Summary

Normalizes user-facing prose across lessons, pages, docs, and Vale rules for consistent Tenstorrent product naming, hardware IDs, trademarks, and terminology. **No functional/code-path changes** — lesson slugs, command IDs, env vars (`MESH_DEVICE=N150`), and executable code blocks are preserved.

### Naming & trademarks

- **TT-Metalium<sup>™</sup>**, **TT-NN<sup>™</sup>**, **TT-Forge<sup>™</sup>**: first mention per page → `<sup>™</sup>` (trademark, not registered)
- **Blackhole**, **TT-QuietBox 2**: first mention → `<sup>®</sup>`
- **Wormhole**: first mention → `<sup>™</sup>`
- **QuietBox** → **TT-QuietBox** / **TT-QuietBox 2** in prose (not `QB2`)
- **open source** two-word form (not `open-source`) in running prose
- Legacy **TT Metal** → **TT-Metalium** in prose/sample output

### Hardware IDs (prose & sample output)

- `n150`, `n300`, `p100`, `p150`, `p300c` — lowercase
- `T3K` → **T3000** in prose (env vars stay `MESH_DEVICE=T3K` where required)
- **Galaxy** capitalized in prose (metadata IDs stay `galaxy`)

### Tooling added/updated

| Script | Purpose |
|--------|---------|
| `scripts/add-tt-product-trademarks.js` | First-mention ™ for TT-Metalium, TT-NN, TT-Forge |
| `scripts/add-blackhole-trademark.js` | First-mention ® for Blackhole |
| `scripts/add-wormhole-trademark.js` | First-mention ™ for Wormhole |
| `scripts/add-quietbox2-trademark.js` | First-mention ® for TT-QuietBox 2 |
| `scripts/normalize-hardware-copy.js` | N150/N300/T3K/P300c/Galaxy prose casing |
| `scripts/normalize-ttnn-copy.js` | TTNN → TT-NN (prose/sample output) |
| `scripts/normalize-tt-metal-copy.js` | TT Metal → TT-Metalium (prose) |
| `scripts/normalize-tt-quietbox-copy.js` | QuietBox → TT-QuietBox |
| `scripts/normalize-open-source-copy.js` | hyphenated form → two-word `open source` |
| `scripts/normalize-tt-product-names-copy.js` | tt-metal / product name normalization |

Also: `docs/STYLE_GUIDE.md`, `styles/Tenstorrent/Terminology.yml`, web catalog hardware labels (`scripts/build-web.js`).

## Intentionally unchanged

- YAML front matter metadata (`supportedHardware: galaxy`, etc.)
- Shell commands, Python/bash code fences, `import ttnn`, `MESH_DEVICE` values
- Lesson slugs, command URIs, repo paths (`~/tt-metal`, `tt-forge-fe`)
- `TTNN` inside Python template docstrings (code, not prose)

## Test plan

- [ ] `npm run build`
- [ ] `npm run package`
- [ ] `node scripts/add-tt-product-trademarks.js --check`
- [ ] `node scripts/normalize-open-source-copy.js --check`
- [ ] `node scripts/normalize-hardware-copy.js --check N150` (and N300, T3K, P300c, galaxy)
- [ ] `node scripts/normalize-ttnn-copy.js --check`
- [ ] F5 Extension Development Host — spot-check walkthrough markdown renders `<sup>™</sup>` / `<sup>®</sup>`
- [ ] Optional: `npm run build:web` → verify `/lessons/` catalog hardware filter labels

## Version

Bumps to **0.0.477** (patch series for copyedit commits).
