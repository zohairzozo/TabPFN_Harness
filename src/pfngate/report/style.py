# ruff: noqa: E501
"""The page's CSS. One inline block, light and dark, no external fonts or files."""

CSS = """
:root {
  --bg: #fcfcfb; --surface: #ffffff; --ink: #0b0b0b; --muted: #52514e; --line: #e4e3df;
  --good: #1a7f4b; --good-bg: #e7f4ec; --bad: #b3261e; --bad-bg: #fbeae8;
  --accent: #2a78d6; --code-bg: #f3f2ef;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121211; --surface: #1b1b1a; --ink: #f2f1ee; --muted: #a9a8a3; --line: #34332f;
    --good: #5fd08f; --good-bg: #16301f; --bad: #ff8a80; --bad-bg: #3a1a17;
    --accent: #6fb0ff; --code-bg: #232321;
  }
}
:root[data-theme="dark"] {
  --bg: #121211; --surface: #1b1b1a; --ink: #f2f1ee; --muted: #a9a8a3; --line: #34332f;
  --good: #5fd08f; --good-bg: #16301f; --bad: #ff8a80; --bad-bg: #3a1a17;
  --accent: #6fb0ff; --code-bg: #232321;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
main { max-width: 1040px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.9rem; margin: 0 0 4px; }
h2 { font-size: 1.35rem; margin: 48px 0 12px; padding-top: 12px; border-top: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 24px 0 8px; }
.secno { color: var(--muted); margin-right: 8px; }
.muted { color: var(--muted); }
nav { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 12px 0 0; font-size: 0.9rem; }
nav a { color: var(--accent); text-decoration: none; }
.banner {
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px 24px;
  padding: 20px; border-radius: 12px; border: 1px solid var(--line); margin: 16px 0;
}
.banner.go { background: var(--good-bg); border-color: var(--good); }
.banner.nogo { background: var(--bad-bg); border-color: var(--bad); }
.banner .word { font-size: 2.6rem; font-weight: 800; letter-spacing: 0.02em; }
.banner.go .word { color: var(--good); }
.banner.nogo .word { color: var(--bad); }
.scroll { overflow-x: auto; margin: 8px 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; background: var(--surface); }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; background: var(--code-bg); }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.group td { background: var(--code-bg); font-weight: 600; }
.pass { color: var(--good); font-weight: 700; }
.fail { color: var(--bad); font-weight: 700; }
figure { margin: 16px 0; }
figure img, .card img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 4px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.card { border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin: 16px 0; background: var(--surface); }
.card h4 { margin: 0 0 4px; font-size: 1rem; }
.card .cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
blockquote { margin: 8px 0; padding: 8px 12px; border-left: 3px solid var(--line); color: var(--muted); }
code, .mono { font: 0.85em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--code-bg); padding: 1px 4px; border-radius: 4px; word-break: break-all; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 8px 0; font-size: 0.9rem; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
details { margin: 8px 0; }
summary { cursor: pointer; }
ul { padding-left: 20px; }
.note { font-size: 0.9rem; color: var(--muted); }
"""
