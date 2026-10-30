"""v2 step 10 - full-text (tier 2) LLM extraction via the Anthropic Message Batches API.

Run on YOUR laptop (ANTHROPIC_API_KEY in the environment):
  python analysis/v2/v2_10_llm_fulltext.py --dry-run                 # count papers + cost estimate, send nothing
  python analysis/v2/v2_10_llm_fulltext.py                           # all usable full texts with Haiku 5.5  -> run 'fulltext_haiku'
  python analysis/v2/v2_10_llm_fulltext.py --models sonnet --sample 40   # validation: same prompt, 40 random papers, Sonnet 5.5
  python analysis/v2/v2_10_llm_fulltext.py fetch <run_name>          # re-download results if the wait was interrupted
  python analysis/v2/v2_10_llm_fulltext.py --all                     # resend everything (normally NOT wanted)

Default: papers that already have a successful result for the same model in ANY earlier fulltext_<model>* run are skipped,
and the new run gets a dated name (fulltext_<model>_add_YYYYMMDD_HHMM) so earlier results are never overwritten.
The analysis combines all runs of a model.

Input: data/processed/v2/fulltext_index.csv (status ok) -> data/processed/v2/fulltext/<id>.txt, truncated to MAX_CHARS
(references already stripped by step 7). Same system prompt and output schema as the abstract run (analysis/v2/llm/).
Output: data/processed/v2/llm/<run>_{requests_map.csv,batch.json,results.jsonl}
"""
import argparse, csv, json, os, random, sys, importlib.util
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('a4', os.path.join(HERE, 'v2_04_llm_abstract.py'))
a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
ROOT = a4.ROOT; IDX = os.path.join(ROOT, 'data', 'processed', 'v2', 'fulltext_index.csv')
MAX_CHARS = 60000       # ~16k tokens; p90 of our texts is below this, longer ones are cut at the end (usually SI / appendices)

def user_message(r, text):
    return (f"FULL-TEXT RUN\n<paper>\n<title>{r['title']}</title>\n<year>{r.get('year', '')}</year>\n<sections>\n{text[:MAX_CHARS]}\n"
            f"</sections>\n</paper>\nThe text was converted from PDF: tables may be flattened and figure content is missing. "
            f"Record only values written in the text, each with a verbatim quote copied from this text.")

def done_uids(name):
    """uids with a successful output in any existing fulltext_<name>* run (results file present)."""
    done = set()
    for f in os.listdir(a4.OUTDIR):
        if f.startswith(f'fulltext_{name}') and f.endswith('_results.jsonl'):
            run = f[:-len('_results.jsonl')]
            mp = os.path.join(a4.OUTDIR, f'{run}_requests_map.csv')
            if not os.path.exists(mp): continue
            m = {r['custom_id']: r['uid'] for r in csv.DictReader(open(mp, encoding='utf-8'))}
            for line in open(os.path.join(a4.OUTDIR, f), encoding='utf-8'):
                r = json.loads(line)
                if r.get('status') == 'succeeded' and r.get('output'): done.add(m[r['custom_id']])
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='run', choices=['run', 'fetch'])
    ap.add_argument('run', nargs='?')
    ap.add_argument('--models', default='haiku'); ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--all', action='store_true', help='do not skip papers that already have results')
    a = ap.parse_args()
    if a.mode == 'fetch':
        info = json.load(open(os.path.join(a4.OUTDIR, f'{a.run}_batch.json'))); a4.wait_and_fetch(a4.client(), a.run, info['batch_id']); return
    corpus = {r['uid']: r for r in csv.DictReader(open(a4.CORPUS, encoding='utf-8'))}
    idx = [r for r in csv.DictReader(open(IDX, encoding='utf-8')) if r['status'] == 'ok']
    if a.sample:
        random.Random(2026).shuffle(idx); idx = idx[:a.sample]
    system, schema = a4.load_system_prompt(), a4.load_tool()
    built, total = [], 0.0
    import time
    for name in a.models.split(','):
        model = a4.MODELS[name]; base = f'fulltext_{name}' + (f'_sample{a.sample}' if a.sample else '')
        done = set() if a.all else done_uids(name)
        todo = [r for r in idx if r['uid'] not in done]
        pending = [f[:-len('_batch.json')] for f in os.listdir(a4.OUTDIR) if f.startswith(f'fulltext_{name}') and f.endswith('_batch.json')
                   and not os.path.exists(os.path.join(a4.OUTDIR, f[:-len('_batch.json')] + '_results.jsonl'))]
        if pending and not a.all:
            sys.exit(f'Run(s) {pending} were submitted but have no results file yet. Wait for them to finish '
                     f'(or: python analysis/v2/v2_10_llm_fulltext.py fetch <run>) before sending more, to avoid paying twice.')
        exists = os.path.exists(os.path.join(a4.OUTDIR, f'{base}_batch.json'))
        run = base if not (done or exists) else f'fulltext_{name}_add_{time.strftime("%Y%m%d_%H%M")}'
        print(f'{name}: {len(idx)} texts, {len(done & {r["uid"] for r in idx})} already done -> {len(todo)} to send as run {run}')
        if not todo: continue
        reqs, mapping = [], []
        for i, r in enumerate(todo):
            text = open(os.path.join(ROOT, *r['file'].replace('\\', '/').split('/')), encoding='utf-8').read()
            cid = f'{run}-{i:05d}'
            reqs.append({'custom_id': cid, 'params': {
                'model': model, 'max_tokens': 16000,
                'system': [{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
                'output_config': {'format': {'type': 'json_schema', 'schema': schema}},
                'messages': [{'role': 'user', 'content': user_message(corpus[r['uid']], text)}]}})
            mapping.append({'custom_id': cid, 'uid': r['uid'], 'gid': '', 'model': model})
        chars = sum(len(q['params']['messages'][0]['content']) for q in reqs)
        pi, po = a4.PRICE[model]; cost = chars / 3.8 / 1e6 * pi + 2500 * len(reqs) / 1e6 * po; total += cost
        print(f'{run}: {len(reqs)} papers, ~{chars / 3.8 / 1e6:.1f}M input tokens, estimated cost <= ${cost:.2f}')
        built.append((run, reqs, mapping))
    print(f'TOTAL estimated <= ${total:.2f}')
    if not built: print('Nothing new to send.'); return
    if a.dry_run: print('Dry run: nothing sent.'); return
    c = a4.client()
    ids = [(run, a4.submit(c, run, reqs, mapping)) for run, reqs, mapping in built]
    for run, bid in ids: a4.wait_and_fetch(c, run, bid)

if __name__ == '__main__':
    main()
