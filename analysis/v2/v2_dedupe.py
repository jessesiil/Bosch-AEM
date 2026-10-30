"""v2 shared helper - one record per paper. Used by v2_08, v2_09, v2_11, v2_12, v2_13, v2_14, v2_15, v2_16
(added 2026-10-22, task T-058; rework the same day after QA: cover records, abstract key, keeper rank).

Why: v2_01_merge.py deduplicates on DOI only, so corpus.csv holds
  (a) the same article under two ids: Angewandte Chemie German edition (10.1002/ange.*) + International edition
      (10.1002/anie.*), repository copies of a publisher article (Zenodo, university repositories, OSTI, Figshare, ...),
      language/translation copies, DOI spelling variants;
  (b) Wiley cover / frontispiece records: a separate DOI whose 'abstract' summarises an article of the same issue.
Rules
  Cover record (dropped, never counted): title starts 'Cover Feature' / 'Frontispiece' / '(Inside/Back/Front) Cover',
      or ends in an issue tag such as '(Adv. Energy Mater. 24/2025)', or the abstract starts 'In article number ...' /
      'In Research Article ...' / 'The Cover Feature ...'.
  Same paper: identical normalised title (lower case, letters+digits, > 30 characters) OR identical normalised first
      200 characters of the abstract (abstracts shorter than that are never used as a key). Groups are transitive.
  Which record is kept (keeper_rank, highest wins): publisher record over repository copy (copy = no DOI, or a
      repository-type source name) > more values that passed the quote checks > International edition (anie) >
      record also in Scopus > alphabetically last uid (determinism only).
"""
import csv, os, re
from collections import defaultdict

MIN_TITLE_CHARS = 30
ABSTRACT_KEY_CHARS = 200
COVER_TITLE = re.compile(r'^\s*(cover feature|frontispiece|(inside |back |front )+cover\b|cover (picture|image|art)\b)', re.I)
ISSUE_TAG = re.compile(r'\((?:[A-Z][\w.&\- ]*\s)?\d{1,2}\s*/\s*(?:19|20)\d\d\)\s*$')
COVER_ABSTRACT = re.compile(r'^\s*(in (research )?article (number )?e?\d|the (cover feature|frontispiece)\b)', re.I)
REPOSITORY = re.compile(r'zenodo|eprints|repositor|publikationsportal|publication server|publications? \(|osti oai|figshare|'
                        r'hochschulbibliothek|open science|depositonce|universit(ä|ae)tsbibliothek|universitätsverlag|repec|kitopen|'
                        r'hispana|funding agency|scholar commons|juser|\bopus\b|\bedoc\b|differ: publications|research collection', re.I)


def norm_title(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())


def title_key(uid, title):
    k = norm_title(title)
    return k if len(k) > MIN_TITLE_CHARS else uid


def abstract_key(abstract):
    k = re.sub(r'[^a-z0-9]', '', (abstract or '').lower())
    return k[:ABSTRACT_KEY_CHARS] if len(k) >= ABSTRACT_KEY_CHARS else None


def is_cover_record(title, abstract=''):
    return bool(COVER_TITLE.search(title or '') or ISSUE_TAG.search(title or '') or COVER_ABSTRACT.search(abstract or ''))


def is_repository_copy(uid, source):
    return not uid.startswith('10.') or bool(REPOSITORY.search(source or ''))


def collapse_duplicate_papers(papers, rank=None):
    """papers: iterable of (uid, title) or (uid, title, abstract); repeats are fine. rank: uid -> tuple, higher = kept.
    Returns (keep, dropped): sets of uids."""
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    uids = set()
    for p in papers:
        uid, title, abstract = (tuple(p) + ('',))[:3]
        uids.add(uid); parent[find(uid)] = find('T:' + title_key(uid, title))
        ak = abstract_key(abstract)
        if ak: parent[find(uid)] = find('A:' + ak)
    groups = defaultdict(set)
    for u in uids: groups[find(u)].add(u)
    rank = rank or (lambda u: ('/anie.' in u,))
    keep = {max(us, key=lambda u: (rank(u), u)) for us in groups.values()}
    return keep, uids - keep


def keeper_rank(corpus, n_kept):
    """Rank for collapse_duplicate_papers. corpus: uid -> corpus row (needs 'source', 'db');
    n_kept: uid -> number of values (performance points + durability results) that passed the quote checks."""
    def r(u):
        c = corpus.get(u, {})
        return (not is_repository_copy(u, c.get('source', '')), n_kept.get(u, 0), '/anie.' in u, c.get('db', '') in ('both', 'scopus'))
    return r


def corpus_texts(root):
    """uid -> (title, abstract, source, db) from data/processed/v2/corpus.csv (for the checks in steps 09-16)."""
    with open(os.path.join(root, 'data', 'processed', 'v2', 'corpus.csv'), encoding='utf-8') as f:
        return {r['uid']: dict(title=r['title'], abstract=r['abstract'], source=r['source'], db=r['db']) for r in csv.DictReader(f)}


def assert_no_duplicate_papers(rows, label, texts):
    """Stop with a clear message if rows (dicts with uid) still contain a cover record or one paper under two uids,
    i.e. the input was made before the fix in v2_08 / v2_11 - re-run those first. texts = corpus_texts(ROOT)."""
    uids = {r['uid'] for r in rows}
    covers = sorted(u for u in uids if is_cover_record(texts[u]['title'], texts[u]['abstract']))
    _, dropped = collapse_duplicate_papers((u, texts[u]['title'], texts[u]['abstract']) for u in uids)
    if covers or dropped:
        raise SystemExit(f'{label}: {len(covers)} cover records, {len(dropped)} duplicate papers (e.g. {(covers + sorted(dropped))[:3]}). '
                         f'Re-run v2_08_clean_tier1.py / v2_11_clean_tier2.py first.')
