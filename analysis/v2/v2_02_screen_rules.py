"""v2 step 2 - transparent rule-based screen. Every rule is a named regex below; the same list is documented in
analysis/v2/SCREENING_RULES.md. All regexes are case-insensitive unless marked CS (case-sensitive).
Stage 1 (EXCLUDE clear junk, tuned for recall) -> Stage 2 (FLAGS used for prioritising and for comparison with the LLM screen).
Output: data/processed/v2/corpus_screened.csv + data/processed/v2/screen_flow.json"""
import csv, re, os, json, collections
csv.field_size_limit(10**9)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
IN = os.path.join(ROOT, 'data/processed/v2/corpus.csv')
OUT = os.path.join(ROOT, 'data/processed/v2/corpus_screened.csv')
FLOW = os.path.join(ROOT, 'data/processed/v2/screen_flow.json')
I = re.I

# ---------- signal regexes ----------
# A: AEM / anion-exchange membrane signal
A_PHRASE = re.compile(r"anion[- ]?exchange[- ](membrane|ionomer|polymer|binder)|hydroxide[- ](exchange|conduct\w*)[- ](membrane|ionomer|polymer)"
                      r"|alkaline (polymer electrolyte|electrolyte membrane|ionomer|anion[- ]exchange)|alkaline membrane[- ](water )?electroly"
                      r"|anion[- ]conduct\w* (membrane|polymer|ionomer)|\bAEMWEs?\b|\bAEMELs?\b|\bHEMWEs?\b|\bAPEWEs?\b", I)
A_ACRONYM_CS = re.compile(r"\bAEMs?\b")   # CS: upper-case acronym only
# W: hydrogen / water electrolysis signal
W = re.compile(r"water electroly[sz]\w*|electroly[sz]er|electrolyser|water[- ]splitting|hydrogen (production|generation)"
               r"|\bH2 (production|generation)|electrolysis cell|\bAEMWE|\bAEMEL|HEMWE|APEWE|\bHER\b.*\bOER\b|\bOER\b.*\bHER\b", I)
# X: off-scope application in the TITLE (other electrochemical systems)
X_TITLE = re.compile(r"\bCO2\b|CO₂|carbon dioxide|CO2RR|\bCORR?\b|fuel cells?\b|flow batter|redox flow|zinc[- ]air|\bZn[- ]air|metal[- ]air|batter(y|ies)"
                     r"|electrodialysis|desalination|deioni[sz]ation|chlor[- ]?alkali|nitrate|nitrite|ammonia|urea|hydrazine|methanol (oxidation|electro)"
                     r"|ethanol oxidation|glycerol|biomass|furfural|\bHMF\b|hydroxymethylfurfural|alcohol oxidation|photo(catalytic|electro\w*|catalyst)"
                     r"|solar[- ]driven|actuator|sensor|chromatograph|wastewater|lithium|electrowinning|organic (electro)?synthesis", I)
X_TITLE_RESCUE = re.compile(r"water electroly|electroly[sz]er|electrolyser|AEMWE|water[- ]splitting", I)  # title also names water electrolysis -> keep
# D: device-level electrolyser test signal (title+abstract)
D = re.compile(r"membrane[- ]electrode[- ]assembl|\bMEAs?\b|single[- ]cell|full[- ]cell|\bstack|zero[- ]gap|catalyst[- ]coated (membrane|substrate)|\bCCM\b|\bCCS\b"
               r"|cell voltage|electroly[sz]er (test|performance|operation|cell|device)|electrolyser (test|performance|operation|cell|device)"
               r"|AEMWE (cell|device|performance|test)|\bin (an? )?(AEM|anion[- ]exchange membrane) (water )?electroly"
               r"|\d+(\.\d+)?\s*A\s*(cm|/\s*cm)|\d(\.\d+)?\s*V\s*(at|@|and|to reach|for)\b|\d+\s*(A|mA)\s*cm.{0,40}\d\.\d+\s*V|\d\.\d+\s*V.{0,40}\d+(\.\d+)?\s*A\s*cm", I)
# H: half-cell / three-electrode signal (abstract)
H = re.compile(r"10\s*mA\s*(cm|/\s*cm)|\bRDE\b|rotating (ring[- ])?disk|three[- ]electrode|\bTafel\b|overpotentials? of", I)
# S: durability / stability test signal
S = re.compile(r"durab\w*|degradation (rate|mechanism)|long[- ]term (stab|operat|test)|stab\w* (test|operation)|\d+\s*(h|hrs?|hours)\b"
               r"|µV\s*/?\s*h|μV\s*/?\s*h|uV\s*/?\s*h|mV\s*/?\s*h|lifetime|accelerated stress", I)
# P: performance / efficiency signal
P = re.compile(r"current density|\d+(\.\d+)?\s*A\s*(cm|/\s*cm)|efficien\w*|cell voltage|polari[sz]ation curve|kWh", I)
# M: modelling / techno-economic / system papers (no own experiment)
M = re.compile(r"\bmodel(l)?ing\b|simulation|\bCFD\b|techno[- ]?economic|life[- ]cycle|\bLCA\b|levelized cost|LCOH|machine learning|density functional|\bDFT\b", I)

def sig(rx, text): return int(bool(rx.search(text)))

flow = collections.OrderedDict(); rows = []
with open(IN, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        t, ab, kw = r['title'], r['abstract'], r['keywords']
        ta = f"{t} {ab}"; tak = f"{t} {ab} {kw}"
        a = sig(A_PHRASE, tak) or sig(A_ACRONYM_CS, tak)
        w = sig(W, tak)
        x = sig(X_TITLE, t) and not sig(X_TITLE_RESCUE, t)
        # ---------- stage 1 exclusion (first matching rule wins) ----------
        if not a:            ex = 'E1_no_AEM_signal'
        elif not w:          ex = 'E2_no_water_electrolysis_signal'
        elif x:              ex = 'E3_offscope_application_in_title'
        else:                ex = ''
        r.update(stage1='exclude' if ex else 'include', exclusion_rule=ex,
                 sig_A=a, sig_W=w, sig_X=int(bool(x)),
                 sig_D=sig(D, ta), sig_H=sig(H, ab), sig_S=sig(S, ta), sig_P=sig(P, ta), sig_M=sig(M, ta),
                 no_abstract=int(not ab.strip()))
        # ---------- stage 2 rule class (only for included records) ----------
        if ex: cls = ''
        elif r['is_review'] == '1': cls = 'review'
        elif r['sig_D']: cls = 'device_likely'
        elif r['sig_H']: cls = 'halfcell_only_likely'
        elif r['sig_M']: cls = 'model_or_system'
        else: cls = 'unclear'
        r['rule_class'] = cls
        rows.append(r)

flow['total'] = len(rows)
for k, v in collections.Counter(r['exclusion_rule'] or 'included' for r in rows).most_common(): flow[k] = v
flow['included_by_class'] = dict(collections.Counter(r['rule_class'] for r in rows if r['stage1'] == 'include'))
flow['included_no_abstract'] = sum(1 for r in rows if r['stage1'] == 'include' and r['no_abstract'])
flow['device_likely_with_S'] = sum(1 for r in rows if r['rule_class'] == 'device_likely' and r['sig_S'])
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(flow, open(FLOW, 'w'), indent=1); print(json.dumps(flow, indent=1))
