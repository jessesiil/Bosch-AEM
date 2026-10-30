"""
v2_17_slide_limitations.py - 16:9 limitations slide (results, not methodology), transparent background.

Left: the model's scope inside the wider decision (excluded dimensions as grey chips).
Right: three short limitation statements and the take-away. Team palette, PT Serif, text >= 18 pt.
Output  figures/v2/slide_limitations.png
Run     python analysis/v2/v2_17_slide_limitations.py   (set PTSERIF_DIR if PT Serif is not installed)
"""
import glob, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT = os.path.join(ROOT, 'figures', 'v2', 'slide_limitations.png')

GREEN, ORANGE, GREY = '#3c8d5a', '#e5772e', '#7a7a7a'   # team palette - swap for the template's hex codes
INK, INK2 = '#1a1a1a', '#5a5a5a'
dirs = [os.environ.get('PTSERIF_DIR', ''), os.path.expanduser('~/fonts'), 'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts')]
files = [f for d in dirs if d and os.path.isdir(d) for pat in ('*PT*Serif*.ttf', 'PTF*.ttf') for f in glob.glob(os.path.join(d, pat))]
for f in files: font_manager.fontManager.addfont(f)
plt.rcParams['font.family'] = 'PT Serif' if files else 'serif'
if not files: print('WARNING: PT Serif not found - using default serif. Set PTSERIF_DIR.')

W, H = 13.333, 7.5
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
def box(x, y, w, h, fc='none', ec='none', lw=0, r=0.18, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls))
def t(x, y, s, size=18, color=INK, bold=False, **k):
    ax.text(x, y, s, fontsize=size, color=color, fontweight='bold' if bold else 'normal', va='center', **k)

t(0.6, 0.72, 'What our results can and cannot tell you', 30, INK, True)

# two limitation bullets, side by side
items = [('Limited, narrow evidence', 'Lab cells, best results, ~5 years of mixed data'),
         ('A model built on assumptions', 'Focused on a subset of the variables')]
for i, (head, line) in enumerate(items):
    x = 0.6 + i * 6.25; y = 1.75
    ax.add_patch(plt.Circle((x + 0.25, y), 0.25, color=ORANGE))
    t(x + 0.25, y + 0.01, str(i + 1), 20, 'white', True, ha='center')
    t(x + 0.7, y, head, 24, INK, True)
    t(x + 0.7, y + 0.52, line, 19, INK2)

# the model block with the six PESTLE factors
BX, BY, BW, BH = 0.6, 3.05, W - 1.2, 2.75
box(BX, BY, BW, BH, ec=GREEN, lw=2.5, r=0.3)
t(BX + 0.4, BY + 0.5, 'Our LCOH model', 26, GREEN, True)
t(BX + 4.1, BY + 0.52, 'which PESTLE factors it includes', 19, INK2)
chips = [('Technological', True), ('Economic', True), ('Political', False), ('Social', False),
         ('Legal', False), ('Environmental', False)]
cw, ch, gap = 1.8, 0.62, 0.12
x0 = BX + (BW - (6 * cw + 5 * gap)) / 2; cy = BY + 1.2
for k, (name, inc) in enumerate(chips):
    x = x0 + k * (cw + gap)
    if inc:
        box(x, cy, cw, ch, fc=GREEN, r=0.31)
        t(x + cw / 2, cy + ch / 2, name, 18, 'white', True, ha='center')
    else:
        box(x, cy, cw, ch, ec='#b5b5b5', lw=1.6, r=0.31)
        t(x + cw / 2, cy + ch / 2, name, 18, '#9a9a9a', ha='center')
        ax.plot([x + 0.18, x + cw - 0.18], [cy + ch / 2] * 2, color=GREY, lw=2.2)
t(x0 + (2 * cw + gap) / 2, cy + ch + 0.5, 'in the model', 19, GREEN, True, ha='center')
t(x0 + 2 * (cw + gap) + (4 * cw + 3 * gap) / 2, cy + ch + 0.5, 'left out to keep the model usable', 19, INK2, ha='center')

# take-away
PW = 6.0
box((W - PW) / 2, 6.2, PW, 0.8, fc=GREEN, r=0.3)
t(W / 2, 6.6, 'A direction, not a forecast', 23, 'white', True, ha='center')
fig.savefig(OUT, dpi=200, transparent=True); plt.close()
print(OUT)
