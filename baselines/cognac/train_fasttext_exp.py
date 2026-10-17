"""
Build a FastText vocabulary for each of the 10 experiments.

The training and test sets of an experiment may come from different context configurations,
so the vocabulary is built from all JSON files of that dataset type.

Usage:
    python train_fasttext_exp.py --exp 1       # build the vocabulary for experiment 1
    python train_fasttext_exp.py --all          # build the vocabularies for all experiments
"""

import os
import sys
import argparse
import pickle
import json
from typing import List
from gensim.models import FastText


EXPERIMENTS = {
    1: {'ds': 'random', 'train_ctx': 'with_context', 'test_ctx': 'with_context'},
    2: {'ds': 'random', 'train_ctx': 'with_context', 'test_ctx': 'without_context'},
    3: {'ds': 'random', 'train_ctx': 'without_context', 'test_ctx': 'with_context'},
    4: {'ds': 'random', 'train_ctx': 'without_context', 'test_ctx': 'without_context'},
    5: {'ds': 'mask_recursive', 'train_ctx': 'with_context', 'test_ctx': 'with_context'},
    6: {'ds': 'mask_recursive', 'train_ctx': 'with_context', 'test_ctx': 'without_context'},
    7: {'ds': 'mask_recursive', 'train_ctx': 'without_context', 'test_ctx': 'with_context'},
    8: {'ds': 'mask_recursive', 'train_ctx': 'without_context', 'test_ctx': 'without_context'},
    9: {'ds': 'original', 'train_ctx': 'without_context', 'test_ctx': 'without_context'},
    10: {'ds': 'original_ctx', 'train_ctx': 'with_context', 'test_ctx': 'with_context'},
}


def is_alpha(string):
    for x in string:
        if 64 < ord(x) < 123:
            continue
        else:
            return False
    return True


def json2corpus(file_path):
    res = []
    with open(file_path, 'r', encoding='utf8') as f:
        for method_json in f.readlines():
            try:
                method_json = json.loads(method_json)
            except:
                continue
            cur_line = method_json[-1]
            for x in method_json[:-1]:
                if is_alpha(x[0]):
                    cur_line.append(x[0])
            res.append(cur_line)
    return res


def save_vocab_weight(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f_vocab, \
         open(os.path.join(output_dir, 'weight.pkl'), 'wb') as f_weight:
        pickle.dump(model.wv.index2word, f_vocab)
        pickle.dump(model.wv.vectors_vocab[:len(model.wv.index2word), :], f_weight)


def build_vocab_for_exp(exp_id: int, base_dir: str):
    """Build the vocabulary for the given experiment."""
    exp = EXPERIMENTS[exp_id]
    ds = exp['ds']
    train_ctx = exp['train_ctx']
    test_ctx = exp['test_ctx']

    # Collect all data files involved in this experiment
    corpus_paths = []
    train_path = os.path.join(base_dir, 'dataset', ds, train_ctx, 'train.json')
    test_path = os.path.join(base_dir, 'dataset', ds, test_ctx, 'test.json')

    if os.path.exists(train_path):
        corpus_paths.append(train_path)
    if os.path.exists(test_path):
        corpus_paths.append(test_path)

    # For cross testing, also include the training-side test file for validation
    train_test_path = os.path.join(base_dir, 'dataset', ds, train_ctx, 'test.json')
    if train_test_path not in corpus_paths and os.path.exists(train_test_path):
        corpus_paths.append(train_test_path)

    print(f"Experiment {exp_id}: {ds} / train={train_ctx}, test={test_ctx}")
    print(f"  corpus files: {corpus_paths}")

    # Build the vocabulary
    v_dim = 128
    window = 8
    min_count = 1

    extra_words = [['<PAD>'] * min_count + ['<START>'] * min_count +
                   ['<EOS>'] * min_count + ['<UNK>'] * min_count]
    model = FastText(size=v_dim, window=window, min_count=min_count)
    model.build_vocab(sentences=extra_words)

    corpus = []
    for cp in corpus_paths:
        corpus += json2corpus(cp)
    model.build_vocab(sentences=corpus, update=True)

    # Write to the experiment-specific directory
    output_dir = os.path.join(base_dir, 'dataset', f'exp{exp_id}_fasttext_vectors')
    save_vocab_weight(model, output_dir)
    print(f"  vocabulary saved to: {output_dir}")
    print(f"  vocabulary size: {len(model.wv.index2word)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build FastText vocabularies for the experiments")
    parser.add_argument('--exp', type=int, choices=range(1, 11),
                        help='experiment id (1-10)')
    parser.add_argument('--all', action='store_true',
                        help='build the vocabularies for all experiments')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.all:
        for exp_id in range(1, 11):
            build_vocab_for_exp(exp_id, base_dir)
    elif args.exp:
        build_vocab_for_exp(args.exp, base_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
