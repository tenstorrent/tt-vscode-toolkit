# Marketplace Publishing & Web Presence Update

**Date:** 2026-04-24
**Status:** Approved

## Goal

The extension is now live on the VS Code Marketplace. Update all user-facing surfaces to reflect this, fix a wrong marketplace URL propagated through the site, and enable automated publishing on every tagged release.

**Marketplace URL:** `https://marketplace.visualstudio.com/items?itemName=Tenstorrent.tt-vscode-toolkit`
**Extension ID:** `Tenstorrent.tt-vscode-toolkit`

---

## Section 1: README

### Badges
Add a VS Code Marketplace version badge next to the existing License and VSCode badges:
```markdown
[![VS Code Marketplace](https://img.shields.io/visual-studio-marketplace/v/Tenstorrent.tt-vscode-toolkit)](https://marketplace.visualstudio.com/items?itemName=Tenstorrent.tt-vscode-toolkit)
```

### Installation section
Restructure the three install options, promoting marketplace to Option 1:

1. **Option 1: VS Code Marketplace (Recommended)**
   - One-liner: `code --install-extension Tenstorrent.tt-vscode-toolkit`
   - Direct link to the marketplace page
2. **Option 2: Install from VSIX** — current content, renumbered
3. **Option 3: Build from Source** — current content, renumbered

### Release notes body in `release.yml`
Update the GitHub Release body text to mention marketplace install first, with VSIX as the manual fallback.

---

## Section 2: Site install page (`site/install/index.html`)

### Hero CTA
Replace the single "Download latest .vsix" primary button with two buttons:
- **Primary (filled teal):** "Install from VS Code Marketplace" → links to marketplace URL
- **Secondary:** "Download .vsix ↓" → existing GitHub releases link

### Install tabs
Add a "Marketplace" tab as the first tab (before VS Code, Cursor/Windsurf, code-server, CLI):
- One-liner: `code --install-extension Tenstorrent.tt-vscode-toolkit` with copy button
- Direct link to the marketplace page

### Wrong URL fix
All lesson command badge links in the built site point to `Tenstorrent.tenstorrent-toolkit` (wrong). Fix to `Tenstorrent.tt-vscode-toolkit`.

The site is generated — find where this URL is defined in the build pipeline (`scripts/build-web.js` or equivalent) and fix it at source so it propagates correctly to all lesson pages on next build.

---

## Section 3: CI Auto-publish (`release.yml`)

### Enable the job
Remove `if: false` from the existing `publish-marketplace` job. The rest of the job is already correct: it builds, packages, and runs `vsce publish` using `VSCE_PAT`.

**Note:** `VSCE_PAT` secret is already configured in the GitHub repo (completed by user prior to implementation).

### Resulting workflow on tagged release
1. Push `v*.*.*` tag → triggers workflow
2. `create-release` job: build → test → package → create GitHub Release with .vsix attached
3. `publish-marketplace` job (runs after `create-release`): build → package → `vsce publish` → extension live on marketplace

---

## Files to Change

| File | Change |
|------|--------|
| `README.md` | Add marketplace badge; restructure install options |
| `site/install/index.html` | Add marketplace hero button; add Marketplace tab |
| `scripts/build-web.js` (or equivalent) | Fix wrong extension ID `tenstorrent-toolkit` → `tt-vscode-toolkit` |
| `.github/workflows/release.yml` | Remove `if: false` from `publish-marketplace` job; update release body text |
| `package.json` | Increment PATCH version |
| `CHANGELOG.md` | Add entry for this release |
