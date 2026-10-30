"""Parse CORDIS bulk project + organisation CSVs (FP7, H2020, Horizon Europe), repairing rows broken by
unescaped ';' in free-text fields, and flag hydrogen/electrolysis candidate projects with transparent regexes.
Outputs: data/processed/cordis_projects_all.parquet-like CSV (subset of columns) and cordis_h2_candidates.csv"""
import csv, io, os, re, zipfile
import pandas as pd
from common import RAW, PROC
csv.field_size_limit(10**9)
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$'); TS = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')

def repair(row, hdr):
    if len(row) == len(hdr):
        return row, False
    t0 = hdr.index('title'); s0 = hdr.index('startDate'); o0 = hdr.index('objective'); c0 = hdr.index('contentUpdateDate')
    sidx = next(i for i in range(t0 + 1, len(row)) if DATE.match(row[i] or ''))
    cidx = max(i for i in range(len(row)) if TS.match(row[i] or ''))
    nstruct = o0 - s0
    out = row[:t0] + [';'.join(row[t0:sidx])] + row[sidx:sidx + nstruct] + [';'.join(row[sidx + nstruct:cidx])]
    tail = row[cidx:]; htail = hdr[c0:]
    if len(tail) > len(htail):  # extra ';' inside keywords
        k = htail.index('keywords'); extra = len(tail) - len(htail)
        tail = tail[:k] + [';'.join(tail[k:k + extra + 1])] + tail[k + extra + 1:]
    out += tail
    assert len(out) == len(hdr), (len(out), len(hdr), row[0])
    return out, True

UNREPAIRED = []
def read_zip(zname, member):
    raw = zipfile.ZipFile(os.path.join(RAW, 'cordis', zname)).open(member).read().decode('utf-8')
    rows = list(csv.reader(io.StringIO(raw), delimiter=';', quotechar='"'))
    hdr = rows[0]; fixed = 0; data = []
    for r in rows[1:]:
        if member == 'project.csv':
            try:
                r, f = repair(r, hdr); fixed += f
            except (StopIteration, ValueError, AssertionError):
                UNREPAIRED.append((zname, r[0] if r else '', len(r)))
                continue
        elif len(r) != len(hdr):
            continue
        data.append(r)
    return pd.DataFrame(data, columns=hdr), fixed

def num(s):
    try: return float(str(s).replace(',', '.'))
    except Exception: return float('nan')

projs, orgs = [], []
for fp, z in [('FP7', 'cordis-fp7projects-csv.zip'), ('H2020', 'cordis-h2020projects-csv.zip'), ('HORIZON', 'cordis-HORIZONprojects-csv.zip')]:
    p, fixed = read_zip(z, 'project.csv'); o, _ = read_zip(z, 'organization.csv')
    p['programme'] = fp; o['programme'] = fp
    print(fp, 'projects', len(p), 'repaired rows', fixed, 'orgs', len(o))
    projs.append(p); orgs.append(o)
print('unrepaired project rows dropped:', len(UNREPAIRED), UNREPAIRED[:10])
P = pd.concat(projs, ignore_index=True); O = pd.concat(orgs, ignore_index=True)
for c in ['totalCost', 'ecMaxContribution']: P[c] = P[c].map(num)
for c in ['ecContribution', 'netEcContribution', 'totalCost']: O[c] = O[c].map(num)
P['start_year'] = pd.to_datetime(P['startDate'], errors='coerce').dt.year
P['text'] = (P['title'].fillna('') + ' || ' + P['objective'].fillna('') + ' || ' + P.get('keywords', pd.Series('', index=P.index)).fillna(''))

H2 = re.compile(r'hydrogen|\bH2\b|electrolys|electrolyz|power[- ]to[- ](gas|x|hydrogen)', re.I)
ELY = re.compile(r'electrolys[ei]s|electrolyz|electrolyser|water splitting|electrolytic hydrogen|electrolysis of water', re.I)
GREEN = re.compile(r'green hydrogen|renewable hydrogen|clean hydrogen|power[- ]to[- ](gas|hydrogen|x)|electrolytic hydrogen|low[- ]carbon hydrogen|hydrogen valley', re.I)
AEM = re.compile(r'anion[- ]exchange membrane|\bAEM(WE|EL|s)?\b|alkaline (polymer|membrane|exchange membrane)|hydroxide[- ](exchange|conducting)|anion[- ]conducting|alkaline ionomer|AEMWE|\bAPE\b electroly', re.I)
TECH = {'PEM': re.compile(r'\bPEM\b|PEMWE|PEMEL|proton exchange membrane|polymer electrolyte membrane', re.I),
        'ALK': re.compile(r'alkaline (water )?electroly', re.I),
        'SOEC': re.compile(r'solid oxide electroly|\bSOEC|\bSOE\b|co-electrolysis|steam electrolysis', re.I),
        'AEM': AEM}
CH2 = re.compile(r'FCH|CLEANH2|JTI-CLEANH2|HYDROGEN', re.I)

P['m_h2'] = P['text'].str.contains(H2); P['m_ely'] = P['text'].str.contains(ELY); P['m_green'] = P['text'].str.contains(GREEN)
P['m_aem'] = P['text'].str.contains(AEM)
for k, rx in TECH.items(): P['tech_' + k] = P['text'].str.contains(rx)
P['clean_h2_partnership'] = P['topics'].fillna('').str.contains(CH2) | P['fundingScheme'].fillna('').str.contains('FCH|JTI', regex=True)
cand = P[(P['m_h2'] & (P['m_ely'] | P['m_green'])) | (P['m_aem'] & P['m_ely'])].copy()
print('candidates', len(cand), 'with electrolysis term', int(cand['m_ely'].sum()), 'AEM term', int(cand['m_aem'].sum()))

# participant mix per project (activityType: PRC private for-profit, HES higher edu, REC research org, PUB public, OTH)
O['projectID'] = O['projectID'].astype(str)
mix = O.groupby(['projectID', 'activityType'])['ecContribution'].sum().unstack(fill_value=0)
mix.columns = ['ec_' + c for c in mix.columns]
cand = cand.merge(mix, left_on='id', right_index=True, how='left')
coord = O[O['role'].str.lower() == 'coordinator'][['projectID', 'name', 'country', 'activityType']].drop_duplicates('projectID')
coord.columns = ['id', 'coord_name', 'coord_country', 'coord_type']
cand = cand.merge(coord, on='id', how='left')
keep = ['id', 'acronym', 'programme', 'title', 'startDate', 'endDate', 'start_year', 'totalCost', 'ecMaxContribution', 'fundingScheme', 'topics', 'clean_h2_partnership',
        'm_ely', 'm_green', 'm_aem'] + ['tech_' + k for k in TECH] + [c for c in cand.columns if c.startswith('ec_')] + ['coord_name', 'coord_country', 'coord_type', 'objective']
os.makedirs(PROC, exist_ok=True)
cand[keep].to_csv(os.path.join(PROC, 'cordis_h2_candidates.csv'), index=False)
P[['id', 'programme', 'start_year', 'ecMaxContribution', 'totalCost', 'fundingScheme']].to_csv(os.path.join(PROC, 'cordis_all_projects_slim.csv'), index=False)
O[O['projectID'].isin(cand['id'])].to_csv(os.path.join(PROC, 'cordis_h2_candidate_orgs.csv'), index=False)
print(cand.groupby('programme').size().to_dict())
