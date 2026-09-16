#!/usr/bin/env python3
"""Generate the project icon and GitHub social-preview card as SVG.

    python3 assets/make_icons.py                  # writes the two .svg masters
    sips -s format png assets/icon.svg --out /tmp/icon-hi.png
    sips -z 512 512 /tmp/icon-hi.png --out assets/icon.png
    sips -s format png assets/social-preview.svg --out assets/social-preview.png
"""

import os

CYAN   = "#22d3ee"   # human mark (matches the game UI)
PINK   = "#f472b6"   # agent mark
GREEN  = "#34d399"   # win / positive value
RED    = "#fb7185"   # negative value
TEXT   = "#e8ecf8"
MUTED  = "#8b97b8"
BG     = "#070b16"

# board layout in a 512-unit space
CELL, GAP = 104, 16
POS = [84, 204, 324]                      # x/y of each cell's top-left
CX  = [p + CELL / 2 for p in POS]         # cell centres: 136, 256, 376

MARKS = {0: 1, 3: 1, 8: 1, 2: 2, 4: 2, 6: 2}      # index -> player (1 = human X, 2 = agent O)
LINE  = [2, 4, 6]                                  # agent wins the anti-diagonal
VALUES = {1: ("+0.8", GREEN), 5: ("0.00", MUTED), 7: ("\u22121.0", RED)}


def board(ox, oy, s, glow=True):
    """SVG for the board, transformed to (ox, oy) with scale s."""
    def X(v): return round(ox + v * s, 2)
    def Y(v): return round(oy + v * s, 2)
    def L(v): return round(v * s, 2)
    out = []

    # cells
    for i in range(9):
        r, c = divmod(i, 3)
        x, y = X(POS[c]), Y(POS[r])
        fill = "rgba(255,255,255,0.05)"
        stroke = "rgba(255,255,255,0.11)"
        if i in VALUES:
            _, col = VALUES[i]
            fill = col.replace("#", "rgba(") if False else fill
        out.append(f'<rect x="{x}" y="{y}" width="{L(CELL)}" height="{L(CELL)}" '
                   f'rx="{L(24)}" fill="{fill}" stroke="{stroke}" stroke-width="{L(2.5)}"/>')

    # value tints in the empty squares (the learned values, RL flavour)
    for i, (_, col) in VALUES.items():
        r, c = divmod(i, 3)
        alpha = 0.17 if col != MUTED else 0.05
        out.append(f'<rect x="{X(POS[c])}" y="{Y(POS[r])}" width="{L(CELL)}" height="{L(CELL)}" '
                   f'rx="{L(24)}" fill="{col}" fill-opacity="{alpha}"/>')

    # winning line: layered strokes fake a glow without SVG filters
    # cell 2 is row 0 / col 2 (top-right), cell 6 is row 2 / col 0 (bottom-left)
    x1, y1, x2, y2 = X(CX[2]), Y(CX[0]), X(CX[0]), Y(CX[2])
    if glow:
        for w, a in ((38, 0.05), (27, 0.09), (19, 0.14), (14, 0.22)):
            out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{GREEN}" '
                       f'stroke-opacity="{a}" stroke-width="{L(w)}" stroke-linecap="round"/>')

    # marks, then the winning line on top so it reads as the finished game
    for i, p in MARKS.items():
        r, c = divmod(i, 3)
        cx, cy = X(CX[c]), Y(CX[r])
        fade = 1.0 if i in LINE else 0.62
        if p == 1:
            d = L(26)
            out.append(f'<g stroke="{CYAN}" stroke-opacity="{fade}" stroke-width="{L(13)}" '
                       f'stroke-linecap="round">'
                       f'<line x1="{round(cx-d,2)}" y1="{round(cy-d,2)}" x2="{round(cx+d,2)}" y2="{round(cy+d,2)}"/>'
                       f'<line x1="{round(cx+d,2)}" y1="{round(cy-d,2)}" x2="{round(cx-d,2)}" y2="{round(cy+d,2)}"/></g>')
        else:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{L(27)}" fill="none" stroke="{PINK}" '
                       f'stroke-opacity="{fade}" stroke-width="{L(13)}"/>')

    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#a7f3d0" '
               f'stroke-opacity="0.95" stroke-width="{L(11)}" stroke-linecap="round"/>')

    # learned values printed in the empty squares
    for i, (txt, col) in VALUES.items():
        r, c = divmod(i, 3)
        out.append(f'<text x="{X(CX[c])}" y="{round(Y(CX[r]) + L(11), 2)}" fill="{col}" '
                   f'fill-opacity="0.85" font-family="Menlo, monospace" font-size="{L(30)}" '
                   f'text-anchor="middle">{txt}</text>')
    return "\n  ".join(out)


DEFS = f'''<defs>
    <radialGradient id="bgGlow" cx="30%" cy="18%" r="85%">
      <stop offset="0" stop-color="#12203c"/><stop offset="1" stop-color="{BG}"/>
    </radialGradient>
    <radialGradient id="cyanHalo" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="{CYAN}" stop-opacity="0.18"/>
      <stop offset="1" stop-color="{CYAN}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="pinkHalo" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="{PINK}" stop-opacity="0.16"/>
      <stop offset="1" stop-color="{PINK}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{CYAN}"/><stop offset="0.55" stop-color="#a78bfa"/>
      <stop offset="1" stop-color="{PINK}"/>
    </linearGradient>
    <linearGradient id="curve" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{RED}"/><stop offset="0.5" stop-color="#fbbf24"/>
      <stop offset="1" stop-color="{GREEN}"/>
    </linearGradient>
  </defs>'''

# ---------------------------------------------------------------- square icon
icon = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 512 512">
  {DEFS}
  <rect width="512" height="512" rx="112" fill="url(#bgGlow)"/>
  <rect width="512" height="512" rx="112" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="2"/>
  <circle cx="150" cy="120" r="200" fill="url(#cyanHalo)"/>
  <circle cx="400" cy="420" r="210" fill="url(#pinkHalo)"/>
  {board(0, 0, 1)}
</svg>
'''

# ------------------------------------------------------------ social preview
W, H = 1280, 640
card = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  {DEFS}
  <rect width="{W}" height="{H}" fill="url(#bgGlow)"/>
  <circle cx="120" cy="60" r="330" fill="url(#cyanHalo)"/>
  <circle cx="1180" cy="620" r="360" fill="url(#pinkHalo)"/>
  {board(56, 76, 0.90)}

  <text x="600" y="196" fill="{TEXT}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="52" font-weight="700">Tic-Tac-Toe vs.</text>
  <text x="600" y="256" fill="url(#accent)" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="52" font-weight="700">a self-play RL agent</text>

  <text x="602" y="318" fill="{MUTED}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="23">400,000 self-play games &#183; 627 states &#183; 8 seconds to train</text>

  <rect x="600" y="342" width="566" height="52" rx="26" fill="{GREEN}" fill-opacity="0.12"
        stroke="{GREEN}" stroke-opacity="0.35"/>
  <text x="628" y="375" fill="#a7f3d0" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="21" font-weight="600">Verified never loses &#8212; exhaustive game-tree search</text>

  <text x="602" y="452" fill="{MUTED}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="18">losing lines found by exhaustive search, as training runs</text>
  <polyline points="602,498 742,524 882,550" fill="none" stroke="url(#curve)" stroke-width="5"
            stroke-linecap="round" stroke-linejoin="round"/>
  <g fill="{GREEN}"><circle cx="882" cy="550" r="9"/></g>
  <g fill="#fbbf24"><circle cx="742" cy="524" r="7"/></g>
  <g fill="{RED}"><circle cx="602" cy="498" r="7"/></g>
  <g font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="20" font-weight="700"
     text-anchor="middle">
    <text x="602" y="482" fill="{RED}">166</text>
    <text x="742" y="508" fill="#fbbf24">81</text>
    <text x="886" y="534" fill="{GREEN}">0</text>
  </g>
  <g fill="{MUTED}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="17"
     text-anchor="middle">
    <text x="602" y="586">1k</text>
    <text x="742" y="586">20k</text>
    <text x="882" y="586">400k</text>
  </g>

  <text x="602" y="614" fill="{MUTED}" font-family="Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="19">github.com/sidhpurwala-huzaifa/RL-tictactoe</text>
</svg>
'''

here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, 'icon.svg'), 'w').write(icon)
open(os.path.join(here, 'social-preview.svg'), 'w').write(card)
print("wrote assets/icon.svg and assets/social-preview.svg")
