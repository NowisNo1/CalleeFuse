"""
Convert JSON data into the pickle format accepted by the GTNM model.
Code tokenization uses the original GTNM Normalizer (tree-sitter).

Two modes are supported:
  --use_children_context: recursively collect child method bodies as the pro context
  without it: pro stays empty (all zeros)

Examples:
  # with context
  python data_processing/convert_json_data.py \
      --input_file train_d_2_b_128_l_16_random_no_test.json \
      --output_prefix data_processed/java-large/train_random_with_ctx_subword \
      --use_children_context

  # without context
  python data_processing/convert_json_data.py \
      --input_file train_d_2_b_128_l_16_mask_recursive_no_test.json \
      --output_prefix data_processed/java-large/train_mask_no_ctx_subword
"""
import json
import pickle
import argparse
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.normalizer import Normalizer
from data_processing.extract_data_subword import load_vocab, data_to_id, str_split


normalizer = Normalizer("java")


def encode_code(code, w2id, end_of_line):
    """Encode code with the original Normalizer + data_to_id."""
    codes = normalizer.normalize(code)
    if not codes or not len(codes):
        return []
    initial_tokens = []
    for code_segment in codes.splitlines():
        initial_tokens.extend(data_to_id(code_segment, w2id))
        initial_tokens.append(end_of_line)
    if code[-1] != "\n":
        initial_tokens.pop(-1)
    return initial_tokens


def extract_method_name(method_full_name):
    """Extract the method name from a fully qualified name."""
    last_dot = method_full_name.rfind('.')
    if last_dot == -1:
        name_with_params = method_full_name
    else:
        name_with_params = method_full_name[last_dot + 1:]
    paren_pos = name_with_params.find('(')
    if paren_pos == -1:
        return name_with_params
    return name_with_params[:paren_pos]


def collect_children_bodies(node):
    """Recursively collect the method_body of all children."""
    bodies = []
    for child in node.get('children', []):
        if 'method_body' in child:
            bodies.append(child['method_body'])
        bodies.extend(collect_children_bodies(child))
    return bodies


def convert(input_file, output_prefix, sub_vocab_file, doc_vocab_file,
            body_context_size, project_context_size, doc_context_size,
            tgt_name_size, use_children_context):

    w2id, id2w = load_vocab(sub_vocab_file)
    END_OF_LINE = w2id.get("<endofline>", 4)

    print(f"Loading JSON: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total samples: {len(data)}")
    print(f"Use children context: {use_children_context}")

    body_samples = []
    pro_samples = []
    doc_samples = []
    tag_samples = []
    skipped = 0

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Converting"):
        root = item['root']
        method_name = extract_method_name(root['method_full_name'])

        # 1. tag: sub-token ids of the method name
        sub_words = str_split(method_name)
        if not sub_words:
            skipped += 1
            continue
        name_tokens = [w2id[sw] if sw in w2id else w2id['UNK'] for sw in sub_words]

        # 2. body: method code (keeps <mask>; tree-sitter parses it leniently)
        body_code = root['method_body']
        body_tokens = encode_code(body_code, w2id, END_OF_LINE)
        if not body_tokens:
            skipped += 1
            continue

        tag_samples.append(name_tokens)
        body_samples.append(body_tokens[-body_context_size:])

        # 3. pro: child method bodies, or empty
        if use_children_context:
            children_bodies = collect_children_bodies(root)
            pro_tokens = []
            for cb in children_bodies:
                cb_tokens = encode_code(cb, w2id, END_OF_LINE)
                if cb_tokens:
                    pro_tokens.extend(cb_tokens)
                    pro_tokens.append(END_OF_LINE)
            pro_samples.append(pro_tokens[-project_context_size:])
        else:
            pro_samples.append([])

        # 4. doc: all zeros
        doc_samples.append([0] * doc_context_size)

    # 5. invoked mask
    print(f"Computing invoked mask... (skipped {skipped} empty samples)")
    invoked_samples = []
    for i in tqdm(range(len(pro_samples)), desc="Invoked mask"):
        if use_children_context and pro_samples[i]:
            body_set = set(body_samples[i])
            invoked = [1 if t in body_set else 0 for t in pro_samples[i]]
        else:
            invoked = []
        invoked_samples.append(invoked)

    # Save
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Saving pickle files...")
    pickle.dump(body_samples, open(output_prefix + "_body.pkl", "wb"))
    pickle.dump(pro_samples, open(output_prefix + "_pro.pkl", "wb"))
    pickle.dump(doc_samples, open(output_prefix + "_doc.pkl", "wb"))
    pickle.dump(tag_samples, open(output_prefix + "_tag.pkl", "wb"))
    pickle.dump(invoked_samples, open(output_prefix + "_invoked.pkl", "wb"))

    print(f"Done! {len(body_samples)} samples -> {output_prefix}_*.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON data to GTNM pickle format")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--sub_vocab_file", type=str, default="data_processing/sub_token_w2id.txt")
    parser.add_argument("--doc_vocab_file", type=str, default="data_processing/doc_w2id.txt")
    parser.add_argument("--body_context_size", type=int, default=55)
    parser.add_argument("--project_context_size", type=int, default=60)
    parser.add_argument("--doc_context_size", type=int, default=10)
    parser.add_argument("--tgt_name_size", type=int, default=5)
    parser.add_argument("--use_children_context", action="store_true",
                        help="Use children method bodies as project context")
    args = parser.parse_args()

    convert(
        args.input_file,
        args.output_prefix,
        args.sub_vocab_file,
        args.doc_vocab_file,
        args.body_context_size,
        args.project_context_size,
        args.doc_context_size,
        args.tgt_name_size,
        args.use_children_context,
    )
