# UI Audit & Change Plan

> Status: **IMPLEMENTED (2026-06-11).** Phases A, B, and most of C are done.
> Also: **d3 vendored locally** (`vendor/d3.min.js`) — fully offline, no CDN.
> Note: the **Galaxies (force) homepage view is disabled** (commented out — it was
> slow); the **Radial** view is the active homepage. Galaxies code stays dormant
> with a pre-warm optimization, ready to re-enable later.
> Produced with the `code-reviewer`, `the-fool`, `feature-forge`/`architecture-designer`
> skill playbooks. Stack stayed **static, `file://`, no framework, no build**.

## Done
- A1 ✅ `viewBox` on all 69 viz SVGs + `svg{height:auto}` → responsive, no clipping.
- A2 ✅ mobile off-canvas drawer + hamburger + scrim (replaces `display:none`).
- A3 ✅ button toggles w/ `aria-expanded`, `:focus-visible` outlines, page title → `<h1>` (60 pages; sidebar title demoted to `<div class="navtitle">`), skip-link, legend → buttons.
- A4 ✅ `prefers-reduced-motion` (CSS + force sim settles & stops).
- B1 ✅ unified accent to blue `#5b9cff` app-wide. B2 ✅ responsive tables. B3 ✅ responsive detail panel. B4 ✅ search count + clear.
- C ✅ skip-link, `aria-hidden` decorative emoji, force pre-warm, **local d3**.
- Deferred: 8 stub pages keep no `<h1>` (placeholders); B5 non-color domain cue; in-notes search.

---
_Original plan below for reference._

## Findings (prioritized)

### 🔴 Critical
- **C1 · Mobile = no navigation.** `notes.css` only has `@media(max-width:820px){nav.sidebar{display:none}}` — below 820px the whole left pane (graph link, outline, connections, local graph) disappears with no replacement.
- **C2 · Vizzes don't scale.** ~62 pages have `<svg width="640">` with **no `viewBox`** → narrow screens clip the viz instead of scaling.

### 🟠 High
- **H1 · Keyboard/a11y.** Sidebar group toggles + legend items are `<div>`s (not focusable, no `aria-expanded`); **0 `:focus` styles** in `notes.css`.
- **H2 · Heading order.** Page title is `<h2>`; only `<h1>` is the sidebar domain → no `<h1>` in `<main>`; broken outline for screen readers.
- **H3 · No `prefers-reduced-motion`.** Force "universe" animates continuously + transitions everywhere.

### 🟡 Medium
- **M1 · Two themes** — homepage (`--bg #070912`, accent blue `#5b9cff`) vs notes (`--bg #0f1117`, accent red `#f87171`).
- **M2 · Search** dims everything, no count/clear; notes pages have no search.
- **M3 · `.cmp` tables** overflow on narrow widths. **M4 · Detail panel** fixed 340px swamps small screens. **M5 · Domains color-only encoded.**

### 🟢 Low
- No skip-to-content link · emoji used as functional icons (🕸 ✦ 📄) read awkwardly by SR · galaxies look unsettled ~1s on load · `d3` loaded from CDN (offline gap).

## The Fool (red-team)
- Force "universe" is nondeterministic → no stable mental map; radial was better for recall. Consider remembering last mode or defaulting to radial.
- Built for desktop, but second-brain lookups happen on phones — and mobile nav is currently broken (C1).
- Accessibility is at zero — fine for solo desktop, a wall otherwise.

## Plan (phases + acceptance criteria)

### Phase A — Correctness & reach (🔴🟠)
- **A1 Responsive vizzes** — add `viewBox="0 0 W H"` to every viz `<svg>` (scriptable sweep) + CSS `svg{height:auto}`. *AC: vizzes scale cleanly to ~360px, no clipping.*
- **A2 Mobile navigation** — replace `display:none` with an off-canvas drawer / hamburger under 820px. *AC: all sidebar nav reachable at 375px.*
- **A3 Accessibility** — real `<button>` toggles with `aria-expanded`; `:focus-visible` outlines; page title becomes the `<h1>`; skip-link. *AC: full keyboard operation, visible focus, one `<h1>` per page.*
- **A4 Reduced motion** — gate the force simulation + transitions behind `prefers-reduced-motion`. *AC: no continuous motion when the setting is on.*

### Phase B — Consistency & polish (🟡)
- B1 unify design tokens (single accent/bg, or intentional per-domain accent; align homepage & notes backgrounds).
- B2 responsive `.cmp` tables (scroll wrapper). B3 responsive detail panel width. B4 search: result count + clear button; consider in-notes search.
- B5 secondary (non-color) domain cue.

### Phase C — Nice-to-haves (🟢)
- skip-link · `aria-hidden` on decorative emoji · pre-warm force ticks before first paint · vendor `d3.min.js` for full offline.

## Notes
- Most of Phase A is shared-file edits (`notes.css`/`notes.js`/`index.html`) + a scripted `viewBox` sweep — high impact, low churn.
- Nothing here requires leaving the static/no-build stack.
