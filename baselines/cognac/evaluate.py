"""
Evaluation script: beam-search decoding on the test set and metric computation.

Metrics:
  - Precision: |predicted subtokens & oracle subtokens| / |predicted subtokens|
  - Recall: |intersection| / |oracle subtokens|
  - F1: 2 * P * R / (P + R)
  - Exact Match: fraction of predictions identical to the oracle

Usage:
    python evaluate.py <model_path>
    python evaluate.py <model_dir>  # evaluate every checkpoint in the directory
    python evaluate.py <model_path> --output results/exp1.json
"""

from __future__ import unicode_literals, print_function, division

import sys
import os
import time
import json
import argparse

import torch
import numpy as np
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from training_ptr_gen.model import Model
from training_ptr_gen.train_util import get_input_from_batch


use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class Evaluator(object):
    def __init__(self, model_file_path, decode_data_path=None):
        # decode_data_path explicitly sets the test set; None falls back to the config (backward compatible)
        self.decode_data_path = decode_data_path or config.decode_data_path
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(self.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)
        self.model = Model(model_file_path, is_eval=True)
        self.model_file_path = model_file_path

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        enc_batch, enc_stmts, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, stmt_feature, encoder_hidden = self.model.encoder(enc_batch, enc_stmts, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN)
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(
                y_t_1, s_t_1,
                encoder_outputs, encoder_feature, stmt_feature, enc_padding_mask, c_t_1,
                extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        return beams_sorted[0]

    def evaluate(self):
        """Run decoding and compute all metrics."""
        start = time.time()
        counter = 0

        # Per-sample metrics
        precisions = []
        recalls = []
        f1s = []
        exact_matches = []

        # Bucket by oracle length
        len_precisions = [[] for _ in range(8)]
        len_recalls = [[] for _ in range(8)]
        len_f1s = [[] for _ in range(8)]
        len_em = [[] for _ in range(8)]

        batch = self.batcher.next_batch()
        while batch is not None:
            best_summary = self.beam_search(batch)

            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                pass

            original_abstract_sents = batch.original_abstracts_sents[0]
            oracle_words = [x for x in original_abstract_sents[0].split(' ') if x.strip() != '']

            counter += 1

            # Compute metrics
            pred_set = set(decoded_words)
            oracle_set = set(oracle_words)
            correct = pred_set & oracle_set

            if len(decoded_words) > 0:
                precision = len(correct) / len(decoded_words)
            else:
                precision = 0.0

            if len(oracle_words) > 0:
                recall = len(correct) / len(oracle_words)
            else:
                recall = 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            em = 1.0 if decoded_words == oracle_words else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            exact_matches.append(em)

            # Bucket by length (1-7; >=8 goes to bucket 8)
            bucket = min(len(oracle_words), 8) - 1
            if bucket >= 0:
                len_precisions[bucket].append(precision)
                len_recalls[bucket].append(recall)
                len_f1s[bucket].append(f1)
                len_em[bucket].append(em)

            if counter % 500 == 0:
                elapsed = time.time() - start
                print(f'[{counter} samples | {elapsed:.0f}s] '
                      f'P={np.mean(precisions):.4f} R={np.mean(recalls):.4f} '
                      f'F1={np.mean(f1s):.4f} EM={np.mean(exact_matches):.4f}')

            batch = self.batcher.next_batch()

        total_time = time.time() - start

        # Aggregate results
        results = {
            'model': self.model_file_path,
            'decode_data': self.decode_data_path,
            'total_samples': counter,
            'time_seconds': round(total_time, 1),
            'overall': {
                'precision': round(float(np.mean(precisions)), 4),
                'recall': round(float(np.mean(recalls)), 4),
                'f1': round(float(np.mean(f1s)), 4),
                'exact_match': round(float(np.mean(exact_matches)), 4),
            },
            'by_length': {}
        }

        for i in range(8):
            label = str(i + 1) if i < 7 else '8+'
            if len(len_precisions[i]) > 0:
                results['by_length'][label] = {
                    'count': len(len_precisions[i]),
                    'precision': round(float(np.mean(len_precisions[i])), 4),
                    'recall': round(float(np.mean(len_recalls[i])), 4),
                    'f1': round(float(np.mean(len_f1s[i])), 4),
                    'exact_match': round(float(np.mean(len_em[i])), 4),
                }

        return results


def print_results(results):
    """Format and print the evaluation results."""
    print('\n' + '=' * 60)
    print(f'Model: {results["model"]}')
    print(f'Test set: {results["decode_data"]}')
    print(f'Samples: {results["total_samples"]} | time: {results["time_seconds"]}s')
    print('=' * 60)

    overall = results['overall']
    print(f'\nOverall metrics:')
    print(f'  Precision:    {overall["precision"]:.4f}')
    print(f'  Recall:       {overall["recall"]:.4f}')
    print(f'  F1:           {overall["f1"]:.4f}')
    print(f'  Exact Match:  {overall["exact_match"]:.4f}')

    print(f'\nMetrics by target-name length:')
    print(f'  {"Length":<6} {"Count":<8} {"Precision":<12} {"Recall":<12} {"F1":<12} {"EM":<12}')
    print(f'  {"-"*6} {"-"*8} {"-"*12} {"-"*12} {"-"*12} {"-"*12}')
    for label, stats in results['by_length'].items():
        print(f'  {label:<6} {stats["count"]:<8} {stats["precision"]:<12.4f} '
              f'{stats["recall"]:<12.4f} {stats["f1"]:<12.4f} {stats["exact_match"]:<12.4f}')

    print()


def main():
    parser = argparse.ArgumentParser(description="Cognac model evaluation")
    parser.add_argument('model_path', help='model checkpoint path or directory')
    parser.add_argument('--output', '-o', default=None,
                        help='output JSON file path')
    parser.add_argument('--decode-data', default=None,
                        help='explicitly set the evaluation test set (overrides config.decode_data_path); '
                             'used for cross experiments: select the checkpoint on the training-side validation set, then test on this target set')
    args = parser.parse_args()

    all_results = []

    if os.path.isdir(args.model_path):
        model_files = sorted([os.path.join(args.model_path, f)
                              for f in os.listdir(args.model_path)
                              if f.startswith('model_')])
        print(f'Found {len(model_files)} checkpoints')
    else:
        model_files = [args.model_path]

    for model_file in model_files:
        print(f'\nEvaluating model: {model_file}')
        evaluator = Evaluator(model_file, decode_data_path=args.decode_data)
        results = evaluator.evaluate()
        print_results(results)
        all_results.append(results)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results if len(all_results) > 1 else all_results[0],
                      f, indent=2, ensure_ascii=False)
        print(f'Results saved to: {args.output}')


if __name__ == '__main__':
    main()
