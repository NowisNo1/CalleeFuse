"""
Extract convergence curves from results/exp{id}_all.json.

After training, run_experiments.sh uses evaluate.py to evaluate every checkpoint
and stores full P/R/F1/EM in exp{id}_all.json. This script turns that into:

  Note (two-stage evaluation): exp{id}_all.json records each checkpoint on the [training-side validation set]
  (used for checkpoint selection), so this curve reflects convergence on the model's own validation set,
  separate from the final number in exp{id}.json (the best checkpoint on the [target test set]).
  For self-test experiments the two sets coincide; for cross experiments, read convergence on the training side and the final number on the target side.


  1. convergence_exp{id}.txt - one line per checkpoint, formatted like GTNM's res.txt,
     to judge convergence (whether F1/EM plateaus with step);
  2. convergence_summary.txt - the 4 metrics of the best checkpoint of every experiment, for filling tables.

Usage:
    python extract_convergence.py                 # process all exp*_all.json under results/
    python extract_convergence.py 9               # process exp9 only
    python extract_convergence.py --results-dir results --out-dir results
"""

import os
import re
import json
import argparse


def parse_step(model_path):
    """Parse the training step from a checkpoint path: .../model_10000_1.9981 -> 10000."""
    base = os.path.basename(str(model_path))
    m = re.search(r'model_(\d+)_', base)
    if m:
        return int(m.group(1))
    # Fallback: take the last integer in the path
    nums = re.findall(r'(\d+)', base)
    return int(nums[0]) if nums else -1


def load_records(path):
    """Read all.json and always return a list (the file may contain a dict or a list)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def extract_one(all_json_path, out_dir):
    """Process one exp{id}_all.json, write convergence_exp{id}.txt, and return the best record."""
    exp_tag = os.path.basename(all_json_path).replace('_all.json', '')
    records = load_records(all_json_path)

    # Sort by step to obtain the convergence curve
    rows = []
    for r in records:
        step = parse_step(r.get('model', ''))
        ov = r.get('overall', {})
        rows.append({
            'step': step,
            'precision': ov.get('precision', 0.0),
            'recall': ov.get('recall', 0.0),
            'f1': ov.get('f1', 0.0),
            'exact_match': ov.get('exact_match', 0.0),
        })
    rows.sort(key=lambda x: x['step'])

    out_path = os.path.join(out_dir, 'convergence_%s.txt' % exp_tag)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# %s convergence curve (one line per checkpoint, sorted by step)\n' % exp_tag)
        f.write('# Convergence check: whether f1 / exact_match plateau with step (stop rising or fall back)\n')
        for x in rows:
            f.write('step {}: precision {:.4f}, recall {:.4f}, f1 {:.4f}, exact_match {:.4f}\n'.format(
                x['step'], x['precision'], x['recall'], x['f1'], x['exact_match']))

    # Find the best (by f1) and check whether it falls on an endpoint of the step sequence (endpoint = possibly not converged)
    best = max(rows, key=lambda x: x['f1']) if rows else None
    at_boundary = False
    if best and len(rows) >= 2:
        at_boundary = best['step'] in (rows[0]['step'], rows[-1]['step'])

    print('[{}] {} checkpoints | best f1={:.4f} @ step {} | EM={:.4f}{}'.format(
        exp_tag, len(rows),
        best['f1'] if best else 0.0,
        best['step'] if best else -1,
        best['exact_match'] if best else 0.0,
        '  [WARN] best at endpoint, possibly not converged' if at_boundary else ''))

    return {
        'exp': exp_tag,
        'n_ckpt': len(rows),
        'best': best,
        'at_boundary': at_boundary,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Extract Cognac convergence curves')
    parser.add_argument('exp_ids', nargs='*', help='process only the given experiment ids (e.g. 9); empty means all')
    parser.add_argument('--results-dir', default='results', help='directory containing exp*_all.json')
    parser.add_argument('--out-dir', default='results', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.exp_ids:
        files = [os.path.join(args.results_dir, 'exp%s_all.json' % e) for e in args.exp_ids]
    else:
        files = sorted(
            os.path.join(args.results_dir, fn)
            for fn in os.listdir(args.results_dir)
            if re.match(r'exp\d+_all\.json$', fn)
        )

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print('No exp*_all.json found; run run_experiments.sh to complete the evaluation first.')
        return

    summaries = []
    for fp in files:
        summaries.append(extract_one(fp, args.out_dir))

    # Summary table
    summaries.sort(key=lambda s: s['exp'])
    summary_path = os.path.join(args.out_dir, 'convergence_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('# Summary of the 4 metrics of the best checkpoint of every experiment\n')
        f.write('# Columns: exp  n_ckpt  best_step  precision  recall  f1  exact_match  convergence\n')
        f.write('%-8s %-8s %-10s %-11s %-9s %-9s %-13s %s\n' % (
            'exp', 'n_ckpt', 'best_step', 'precision', 'recall', 'f1', 'exact_match', 'note'))
        for s in summaries:
            b = s['best'] or {}
            note = 'WARN best at endpoint (possibly not converged)' if s['at_boundary'] else 'OK best in the middle (converged)'
            f.write('%-8s %-8d %-10s %-11.4f %-9.4f %-9.4f %-13.4f %s\n' % (
                s['exp'], s['n_ckpt'], str(b.get('step', -1)),
                b.get('precision', 0.0), b.get('recall', 0.0),
                b.get('f1', 0.0), b.get('exact_match', 0.0), note))

    print('\nConvergence curves -> %s/convergence_exp*.txt' % args.out_dir)
    print('Summary            -> %s' % summary_path)


if __name__ == '__main__':
    main()
