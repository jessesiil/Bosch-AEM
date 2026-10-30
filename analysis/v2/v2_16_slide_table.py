"""
v2_16_slide_table.py - 16:9 slide table: 2026 median lab results vs targets, per hurdle.

Rows: peak current density (j at 2.0 V), current density held in durability tests,
efficiency (cell voltage at 1 A/cm2, as kWh/kg), durability test length, degradation rate
(tests >= 1000 h). Status pill colours follow the team palette (green / orange / grey).
Presentation style: PT Serif, every text >= 18 pt when the image fills a 13.33 x 7.5 in slide.

Inputs  data/processed/v2/tier1/perf_points.csv, data/processed/v2/tables/durability_tests.csv (v2_14)
Output  figures/v2/slide_table_2026.png
Run     python analysis/v2/v2_16_slide_table.py
Font    looks for PT Serif in $PTSERIF_DIR, ~/fonts, and the Windows font folders; falls back
        to the default serif with a warning. Free download: https://fonts.google.com/specimen/PT+Serif
"""
import csv, glob, os, re, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
csv.field_size_limit(10**9)
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
P = os.path.join(ROOT, 'data', 'processed', 'v2'); OUT = os.path.join(ROOT, 'figures', 'v2', 'slide_table_2026.png')
YEAR = '2026'

# team palette - replace with the exact hex codes from the presentation template
GREEN, ORANGE, GREY = '#3c8d5a', '#e5772e', '#7a7a7a'
INK, INK2, RULE, BG = '#1a1a1a', '#5a5a5a', '#d6d6d6', '#ffffff'

# ---- font
dirs = [os.environ.get('PTSERIF_DIR', ''), os.path.expanduser('~/fonts'), 'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts')]
files = [f for d in dirs if d and os.path.isdir(d) for pat in ('*PT*Serif*.ttf', 'PTF*.ttf') for f in glob.glob(os.path.join(d, pat))]
for f in files: font_manager.fontManager.addfont(f)
if any('Serif' in font_manager.FontProperties(fname=f).get_name() for f in files):
    plt.rcParams['font.family'] = 'PT Serif'
else:
    print('WARNING: PT Serif not found - using default serif. Set PTSERIF_DIR to the folder with the .ttf files.')
    plt.rcParams['font.family'] = 'serif'

# ---- numbers
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
pp = list(csv.DictReader(open(os.path.join(P, 'tier1', 'perf_points.csv'), encoding='utf-8')))
from v2_dedupe import collapse_duplicate_papers, corpus_texts  # one uid per paper (safety net; v2_08 already collapses duplicates)
_texts = corpus_texts(ROOT)
keep, _ = collapse_duplicate_papers((u, _texts[u]['title'], _texts[u]['abstract']) for u in {r['uid'] for r in pp})
pp = [r for r in pp if r['uid'] in keep and r['year'] == YEAR]
du = [r for r in csv.DictReader(open(os.path.join(P, 'tables', 'durability_tests.csv'), encoding='utf-8-sig'))
      if r['Quote check'] != 'numbers_nearby' and r['Year'] == YEAR]
j2 = st.median([f(r['j_A_cm2']) for r in pp if abs(f(r['V_cell']) - 2.0) <= 0.03])
v1 = st.median([f(r['V_cell']) for r in pp if abs(f(r['j_A_cm2']) - 1.0) <= 0.05])
jh = st.median([f(r['j held (A/cm²)']) for r in du if f(r['j held (A/cm²)'])])
dur = st.median([f(r['Duration (h)']) for r in du if f(r['Duration (h)'])])
rate = st.median([f(r['Rate (µV/h)']) for r in du if f(r['Rate (µV/h)']) and f(r['Rate (µV/h)']) > 0 and (f(r['Duration (h)']) or 0) >= 1000])
kwh = 39.4 * v1 / 1.481     # HHV 39.4 kWh/kg at 1.481 V
print(f'j@2V {j2:.2f}  V@1A {v1:.3f} ({kwh:.1f} kWh/kg)  j held {jh:.2f}  test length {dur:.0f} h  rate(>=1000h) {rate:.1f} uV/h')

ROWS = [  # hurdle, (value, value note), (target, target note), (status, colour)
    ('Peak current density', (f'{j2:.1f} A/cm²', 'at 2.0 V'), ('1.5 A/cm²', 'EU 2030 target'), ('Met in lab', GREEN)),
    ('Current density in\ndurability tests', (f'{jh:.1f} A/cm²', 'held during the test'), ('1.5 A/cm²', 'EU 2030 target'), ('Below target', ORANGE)),
    ('Efficiency', (f'{v1:.2f} V at 1 A/cm²', f'≈ {kwh:.0f} kWh/kg, cell only'), (f'48 kWh/kg', 'EU 2030, whole system'), ('Close', GREY)),
    ('Durability test length', (f'{dur:,.0f} h', 'median test'), ('~80,000 h', 'commercial stack life'), (f'~{80000 / dur:,.0f}× too short', ORANGE)),
    ('Degradation rate', (f'{rate:.0f} µV/h', 'tests of 1,000 h or more'), ('9.5 µV/h', 'EU 2030 target'), (f'~{rate / 9.5:.0f}× too high', ORANGE)),
]

W, H = 13.333, 7.5
fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
X = [0.6, 4.2, 7.5, 10.33]; PILL_W = 2.4
ax.text(X[0], 0.72, 'AEM in 2026 papers: performance is there, durability is not', fontsize=30, fontweight='bold', color=INK, va='center')
y = 1.55
for x, h in zip(X, ['Hurdle', 'Median, 2026 papers', 'Target', 'Status']):
    ax.text(x, y, h, fontsize=20, fontweight='bold', color=INK2, va='center')
ax.plot([X[0], W - 0.6], [y + 0.32, y + 0.32], color=INK2, lw=1.5)
RH = 0.98; y0 = y + 0.32
for i, (hurdle, (v, vn), (t, tn), (s, col)) in enumerate(ROWS):
    yc = y0 + RH * (i + 0.5)
    ax.text(X[0], yc, hurdle, fontsize=21, fontweight='bold', color=INK, va='center', linespacing=1.1)
    for x, a, b in [(X[1], v, vn), (X[2], t, tn)]:
        ax.text(x, yc - 0.17, a, fontsize=24, color=INK, va='center')
        ax.text(x, yc + 0.22, b, fontsize=18, color=INK2, va='center')
    ax.add_patch(FancyBboxPatch((X[3], yc - 0.26), PILL_W, 0.52, boxstyle='round,pad=0,rounding_size=0.26', fc=col, ec='none'))
    ax.text(X[3] + PILL_W / 2, yc, s, fontsize=19, fontweight='bold', color='white', ha='center', va='center')
    if i < len(ROWS) - 1: ax.plot([X[0], W - 0.6], [y0 + RH * (i + 1)] * 2, color=RULE, lw=1)
ax.text(X[0], H - 0.42, 'Lab single cells, mostly in KOH electrolyte; targets apply to commercial stacks and systems.',
        fontsize=18, color=INK2, va='center')
fig.savefig(OUT, dpi=200, transparent=True); plt.close()  # transparent background for the slide
print(OUT)
