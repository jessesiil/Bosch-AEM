"""v2 step 3 - draw the blind gold-labelling sample (seed 2026).
120 from stage-1 INCLUDED (stratified by rule_class) + 30 from stage-1 EXCLUDED (10 per exclusion rule).
Labellers see only title/abstract/keywords in shuffled order; the stratum key is stored separately.
Outputs: data/labels/v2/gold_sample_to_label.xlsx (for labelling), data/labels/v2/gold_sample_key.csv (do not open while labelling)"""
import csv, random, os, collections
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
csv.field_size_limit(10**9)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
rows = list(csv.DictReader(open(os.path.join(ROOT, 'data/processed/v2/corpus_screened.csv'), encoding='utf-8')))
rnd = random.Random(2026)
ALLOC = {'device_likely': 50, 'unclear': 25, 'halfcell_only_likely': 20, 'review': 15, 'model_or_system': 10,
         'E1_no_AEM_signal': 10, 'E2_no_water_electrolysis_signal': 10, 'E3_offscope_application_in_title': 10}
pool = collections.defaultdict(list)
for r in rows: pool[r['rule_class'] or r['exclusion_rule']].append(r)
sample = []
for stratum, n in ALLOC.items():
    for r in rnd.sample(pool[stratum], n):
        sample.append(dict(r, stratum=stratum, stratum_size=len(pool[stratum])))
rnd.shuffle(sample)
for i, r in enumerate(sample, 1): r['gid'] = f'G{i:03d}'

with open(os.path.join(ROOT, 'data/labels/v2/gold_sample_key.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['gid', 'uid', 'stratum', 'stratum_size'])
    for r in sample: w.writerow([r['gid'], r['uid'], r['stratum'], r['stratum_size']])

wb = Workbook(); ws = wb.active; ws.title = 'label'
hdr = ['gid', 'labeller', 'in_scope (Y/N)', 'test_level (device/halfcell/none/unclear)', 'durability_data (Y/N)', 'notes',
       'year', 'title', 'abstract', 'keywords', 'doi']
ws.append(hdr)
for r in sample:
    ws.append([r['gid'], '', '', '', '', '', r['year'], r['title'], r['abstract'] or '[NO ABSTRACT - judge from title; mark unclear if needed]',
               r['keywords'][:400], r['doi']])
widths = [7, 10, 10, 14, 12, 25, 6, 45, 110, 30, 25]
for i, wdt in enumerate(widths): ws.column_dimensions[chr(65 + i)].width = wdt
for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill('solid', fgColor='DDDDDD')
for row in ws.iter_rows(min_row=2):
    for c in row: c.alignment = Alignment(wrap_text=True, vertical='top')
ws.freeze_panes = 'C2'
for col, opts in [('C', '"Y,N"'), ('D', '"device,halfcell,none,unclear"'), ('E', '"Y,N"')]:
    dv = DataValidation(type='list', formula1=opts, allow_blank=True); ws.add_data_validation(dv); dv.add(f'{col}2:{col}{len(sample)+1}')
wb.save(os.path.join(ROOT, 'data/labels/v2/gold_sample_to_label.xlsx'))
print(len(sample), collections.Counter(r['stratum'] for r in sample))
