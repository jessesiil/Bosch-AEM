"""v2 step 4 - abstract-level LLM screening + extraction via the Anthropic Message Batches API.

Run on YOUR computer (the API key is read from the ANTHROPIC_API_KEY environment variable; it is never written to disk).

  python analysis/v2/v2_04_llm_abstract.py gold   --dry-run     # build requests, show cost estimate, send nothing
  python analysis/v2/v2_04_llm_abstract.py gold                 # 150 gold papers x (Haiku 5.5 + Sonnet 5.5)
  python analysis/v2/v2_04_llm_abstract.py corpus --yes         # all stage-1-included papers x Haiku 5.5
  python analysis/v2/v2_04_llm_abstract.py fetch  <run_name>    # re-download results of a submitted run (if the wait was interrupted)

Outputs: data/processed/v2/llm/<run_name>_requests_map.csv, <run_name>_batch.json, <run_name>_results.jsonl
"""
import argparse, csv, json, os, re, sys, time
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
CORPUS = os.path.join(ROOT, 'data', 'processed', 'v2', 'corpus_screened.csv')
GOLD = os.path.join(ROOT, 'data', 'labels', 'v2', 'gold_labels_v2.csv')
PROMPT = os.path.join(HERE, 'llm', 'extraction_prompt.md')
SCHEMA = os.path.join(HERE, 'llm', 'extraction_schema.json')
OUTDIR = os.path.join(ROOT, 'data', 'processed', 'v2', 'llm')

MODELS = {'haiku': 'claude-haiku-5-5', 'sonnet': 'claude-sonnet-5-5'}
# USD per million tokens, Batch API (50% of standard), from platform.claude.com/docs/en/about-claude/pricing (checked 2026-10-15)
PRICE = {'claude-haiku-5-5': (0.05, 0.25), 'claude-sonnet-5-5': (1.0, 5.0)}
MAX_TOKENS = 4000

def load_system_prompt():
    txt = open(PROMPT, encoding='utf-8').read()
    body = txt.split('---')[0]                      # part before the user-message templates
    return body.replace('# System prompt — AEM extraction (used for both abstract-level and full-text runs)', '').strip()

def load_tool():
    # JSON schema for structured output (output_config.format); 'tool' name kept for minimal changes
    return json.load(open(SCHEMA, encoding='utf-8'))

def user_message(r):
    ab = (r.get('abstract') or '').strip()
    note = ('Only the abstract is available. Fill what the abstract states. For highest_test_level: if the abstract reports '
            'any result obtained in an AEM water electrolyser / AEMWE / electrolyzer cell / MEA / single cell, that IS a '
            'single_cell test (use stack only if a stack is stated), even if no cell size or conditions are given. Use '
            'half_cell_only if only three-electrode/overpotential results are reported, and unclear ONLY if the abstract '
            'gives no indication at all of how the work was tested.')
    if not ab:
        ab = '[NO ABSTRACT AVAILABLE - judge from title and keywords only; use unclear where needed]'
    return (f"ABSTRACT-LEVEL RUN\n<paper>\n<title>{r['title']}</title>\n<year>{r.get('year','')}</year>\n"
            f"<abstract>{ab}</abstract>\n<keywords>{(r.get('keywords') or '')[:600]}</keywords>\n</paper>\n{note}")

def build_requests(rows, model, run, system, tool):
    reqs, mapping = [], []
    for i, r in enumerate(rows):
        cid = f"{run}-{i:05d}"
        reqs.append({'custom_id': cid, 'params': {
            'model': model, 'max_tokens': MAX_TOKENS,
            'system': [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
            'output_config': {'format': {'type': 'json_schema', 'schema': tool}},
            'messages': [{'role': 'user', 'content': user_message(r)}]}})
        mapping.append({'custom_id': cid, 'uid': r['uid'], 'gid': r.get('gid', ''), 'model': model})
    return reqs, mapping

def estimate(reqs, model, system, tool):
    fixed = len(system) + len(json.dumps(tool))
    chars_in = sum(fixed + len(q['params']['messages'][0]['content']) for q in reqs)
    tok_in = chars_in / 3.5                     # rough chars->tokens; ignores caching (so this is an upper bound)
    tok_out = 450 * len(reqs)
    pi, po = PRICE[model]
    return tok_in / 1e6 * pi + tok_out / 1e6 * po

def client():
    if not os.environ.get('ANTHROPIC_API_KEY'):
        sys.exit('ANTHROPIC_API_KEY is not set in this terminal. Run: setx ANTHROPIC_API_KEY "<key>"  then open a NEW terminal.')
    try:
        import anthropic
    except ImportError:
        sys.exit('Missing package. Run: python -m pip install anthropic')
    return anthropic.Anthropic()

def submit(c, run, reqs, mapping):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f'{run}_requests_map.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(mapping[0])); w.writeheader(); w.writerows(mapping)
    batch = c.messages.batches.create(requests=reqs)
    json.dump({'run': run, 'batch_id': batch.id, 'n': len(reqs), 'created': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(os.path.join(OUTDIR, f'{run}_batch.json'), 'w'), indent=1)
    print(f'[{run}] submitted batch {batch.id} with {len(reqs)} requests')
    return batch.id

def wait_and_fetch(c, run, batch_id):
    while True:
        b = c.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f'[{run}] {b.processing_status}: done {rc.succeeded}, errored {rc.errored}, processing {rc.processing}', flush=True)
        if b.processing_status == 'ended': break
        time.sleep(30)
    out = os.path.join(OUTDIR, f'{run}_results.jsonl')
    n_ok = n_err = 0; usage_in = usage_out = cache_read = 0
    with open(out, 'w', encoding='utf-8') as f:
        for res in c.messages.batches.results(batch_id):
            rec = {'custom_id': res.custom_id, 'status': res.result.type}
            if res.result.type == 'succeeded':
                m = res.result.message
                txt = ''.join(blk.text for blk in m.content if blk.type == 'text')
                try:
                    rec['output'] = json.loads(txt)
                except Exception:
                    rec['output'] = None; rec['raw_text'] = txt[:2000]
                rec['stop_reason'] = m.stop_reason
                rec['usage'] = {'in': m.usage.input_tokens, 'out': m.usage.output_tokens,
                                'cache_read': getattr(m.usage, 'cache_read_input_tokens', 0) or 0}
                usage_in += rec['usage']['in']; usage_out += rec['usage']['out']; cache_read += rec['usage']['cache_read']
                n_ok += 1
            else:
                rec['error'] = str(getattr(res.result, 'error', res.result.type)); n_err += 1
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'[{run}] saved {out}: {n_ok} ok, {n_err} errors; tokens in {usage_in:,} (+{cache_read:,} cached), out {usage_out:,}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['gold', 'corpus', 'fetch'])
    ap.add_argument('run', nargs='?', help='run name (fetch mode)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--yes', action='store_true', help='confirm the corpus run')
    ap.add_argument('--models', default='', help='comma list: haiku,sonnet (default: gold=both, corpus=haiku)')
    ap.add_argument('--tag', default='', help='suffix for run names, e.g. v2')
    a = ap.parse_args()
    system, tool = load_system_prompt(), load_tool()

    if a.mode == 'fetch':
        info = json.load(open(os.path.join(OUTDIR, f'{a.run}_batch.json')))
        wait_and_fetch(client(), a.run, info['batch_id']); return

    corpus = {r['uid']: r for r in csv.DictReader(open(CORPUS, encoding='utf-8'))}
    if a.mode == 'gold':
        gold = list(csv.DictReader(open(GOLD, encoding='utf-8')))
        rows = [dict(corpus[g['uid']], gid=g['gid']) for g in gold]
        names = a.models.split(',') if a.models else ['haiku', 'sonnet']
    else:
        rows = [r for r in corpus.values() if r['stage1'] == 'include']
        names = a.models.split(',') if a.models else ['haiku']
    sfx = f'_{a.tag}' if a.tag else ''
    plan = [(f'{a.mode}_{n}{sfx}', MODELS[n]) for n in names]

    built, total = [], 0.0
    for run, model in plan:
        reqs, mapping = build_requests(rows, model, run, system, tool)
        cost = estimate(reqs, model, system, tool); total += cost
        print(f'{run}: {len(reqs)} requests on {model}, estimated cost <= ${cost:.2f}')
        built.append((run, reqs, mapping))
    print(f'TOTAL estimated <= ${total:.2f}')
    if a.dry_run:
        print('Dry run: nothing sent.'); return
    if a.mode == 'corpus' and not a.yes:
        sys.exit('Add --yes to confirm the corpus run.')
    c = client()
    ids = [(run, submit(c, run, reqs, mapping)) for run, reqs, mapping in built]   # submit everything first
    for run, bid in ids:
        wait_and_fetch(c, run, bid)

if __name__ == '__main__':
    main()
