"""
Convert the mask_recursive / random / original datasets under raw_data/ into Cognac input format.
Uses javalang for AST parsing and token extraction, consistent with Cognac's dataextractor.py.

Requires javalang (the Ringbo/javalang fork).

Usage:
    python convert_dataset.py --type random --context with
    python convert_dataset.py --type mask_recursive --context without
    python convert_dataset.py --type original --context without
    python convert_dataset.py --type all --context all
    python convert_dataset.py --generate-all   # generate the data for experiments 1-9
    python convert_dataset.py --type original_ctx --context with   # experiment 10
"""

import json
import re
import os
import sys
import argparse
from typing import List, Tuple, Optional

import javalang
import javalang.ast

# Cognac statement-type mapping (same as dataextractor.py)
orederMapping = {
    'StatementExpression': 1,
    'LocalVariableDeclaration': 2,
    'AssertStatement': 3,
    'WhileStatement': 4,
    'IfStatement': 5,
    'TryStatement': 6,
    'ThrowStatement': 7,
    'SwitchStatement': 8,
    'SwitchStatementCase': 9,
    'ReturnStatement': 10,
    'DoStatement': 11,
    'ForStatement': 12,
    'FieldDeclaration': 13,
    'ClassName': 14,
    'ReturnType': 15,
    'Caller': 16,
    'Callee': 17,
    'SynchronizedStatement': 18,
    'Parameter': 19,
    'Callee.StatementExpression': 20,
    'Callee.LocalVariableDeclaration': 21,
    'Callee.AssertStatement': 22,
    'Callee.WhileStatement': 23,
    'Callee.IfStatement': 24,
    'Callee.TryStatement': 25,
    'Callee.ThrowStatement': 26,
    'Callee.SwitchStatement': 27,
    'Callee.SwitchStatementCase': 28,
    'Callee.ReturnStatement': 29,
    'Callee.DoStatement': 30,
    'Callee.ForStatement': 31,
    'Callee.FieldDeclaration': 32,
    'Callee.ClassName': 33,
    'Callee.ReturnType': 34,
    'Callee.SynchronizedStatement': 35,
    'Callee.Parameter': 36,
    'Caller.ReturnType': 37,
    'Caller.Parameter': 38,
    '<UNK>': 39
}

frequence = {
    "ReturnStatement": 0.224632945,
    "ReturnType": 0.160275608,
    "Parameter": 0.125718472,
    "StatementExpression": 0.125082106,
    "LocalVariableDeclaration": 0.099483322,
    "ClassName": 0.096258274,
    "AssertStatement": 0.079472415,
    "SwitchStatement": 0.078355678,
    "SynchronizedStatement": 0.075410688,
    "IfStatement": 0.068725314,
    "ForStatement": 0.062350527,
    "ThrowStatement": 0.055662937,
    "FieldDeclaration": 0.045173916,
    "WhileStatement": 0.038646218,
    "DoStatement": 0.03426961,
    "SwitchStatementCase": 0.033224994,
    "TryStatement": 0.013649739,
}

stripSymbol = re.compile(r'\W+')


def get_subtokens(name: str) -> List[str]:
    """Split camelCase, consistent with Cognac's get_subtokens."""
    return [x.lower() for x in re.split(r'[_ \d]',
            re.sub(r'([A-Z]+[a-z])', r' \1',
            re.sub(r'([A-Z]+)', r' \1', name)).strip()) if x != '']


def is_valid_subtoken(token: str) -> bool:
    """Same filter as dataextractor.py: subtoken.isalpha() and len(strip) > 1"""
    return token.isalpha() and len(stripSymbol.sub('', token)) > 1


def extract_class_name(method_full_name: str) -> str:
    """Extract the class name from method_full_name."""
    parts = method_full_name.split('(')[0].split('.')
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def extract_method_name(method_full_name: str) -> str:
    """Extract the method name from method_full_name."""
    return method_full_name.split('(')[0].split('.')[-1]


def wrap_and_parse(method_body: str):
    """
    Wrap method_body in a complete Java class and parse it with javalang.
    Returns (AST, method_node) or (None, None).
    """
    # Replace <mask> and <extra_id_N> with valid placeholder method names
    body = re.sub(r'<mask>', 'placeholderMethodName', method_body)
    body = re.sub(r'<extra_id_(\d+)>', r'placeholderCallee\1', body)

    wrapped = 'class DummyClass {\n' + body + '\n}'
    try:
        tree = javalang.parse.parse(wrapped)
        if tree.types and tree.types[0].body:
            for member in tree.types[0].body:
                if isinstance(member, (javalang.tree.MethodDeclaration,
                                       javalang.tree.ConstructorDeclaration)):
                    member.tokens = tree.tokens
                    return tree, member
    except:
        pass
    return None, None


def get_token_stream(method_node, topk=0):
    """
    Extract the token stream with javalang.ast.get_token_stream_2.
    Returns a list [(value, curType), ...].
    """
    try:
        token_stream = [x for x in javalang.ast.get_token_stream_2(method_node)
                        if x.value != '' and x.value != ' '
                        and len(x.value.strip()) >= 1
                        and x.curType != "<UNK>"]
    except:
        return []

    if topk == 0:
        return token_stream
    else:
        return sorted(token_stream,
                      key=lambda x: frequence.get('Callee.' + x.curType,
                                                  frequence.get(x.curType, 0.01)),
                      reverse=True)[:topk]


def convert_sample(sample: dict, with_context: bool = True) -> Optional[List]:
    """
    Convert a single sample to Cognac format.
    with_context: True includes callee context; False uses only the method body.
    """
    root = sample['root']
    method_full_name = root['method_full_name']
    method_body = root['method_body']
    children = root.get('children', [])

    # Extract the oracle
    oracle_name = extract_method_name(method_full_name)
    oracle = get_subtokens(oracle_name)
    if len(oracle) == 0:
        return None

    # Parse the root method
    tree, method_node = wrap_and_parse(method_body)
    if method_node is None:
        return None

    proxy = []

    # 1. ClassName
    class_name = extract_class_name(method_full_name)
    proxy.append([class_name, orederMapping['ClassName']])

    # 2. ReturnType
    if hasattr(method_node, 'return_type'):
        if method_node.return_type:
            proxy.append([method_node.return_type.name, orederMapping['ReturnType']])
            if hasattr(method_node.return_type, 'dimensions') and method_node.return_type.dimensions:
                proxy.append(['array', orederMapping['ReturnType']])
        else:
            proxy.append(['void', orederMapping['ReturnType']])

    # 3. Parameters
    if method_node.parameters:
        for parameter in method_node.parameters:
            try:
                type_name = str(parameter.type).split("name='")[1].split("'")[0]
                proxy.append([type_name, orederMapping['Parameter']])
            except:
                pass

    # 4. Callee information (only when with_context=True)
    if with_context:
        for child in children:
            child_body = child.get('method_body', '')
            _, child_node = wrap_and_parse(child_body)
            if child_node is None:
                continue

            # Callee ReturnType
            if hasattr(child_node, 'return_type'):
                if child_node.return_type:
                    proxy.append([child_node.return_type.name, orederMapping['Callee.ReturnType']])
                else:
                    proxy.append(['void', orederMapping['Callee.ReturnType']])

            # Callee Parameters
            if child_node.parameters:
                for parameter in child_node.parameters:
                    try:
                        type_name = str(parameter.type).split("name='")[1].split("'")[0]
                        proxy.append([type_name, orederMapping['Callee.Parameter']])
                    except:
                        pass

            # Callee body tokens (top-10 by frequency)
            callee_tokens = get_token_stream(child_node, topk=10)
            for x in callee_tokens:
                if x.curType != '<UNK>':
                    key = 'Callee.' + x.curType
                    type_id = orederMapping.get(key, orederMapping['<UNK>'])
                else:
                    type_id = orederMapping['<UNK>']
                proxy.append([x.value, type_id])

    # 5. Method body token sequence
    tokens = get_token_stream(method_node)
    if len(tokens) < 3 or len(tokens) < len(oracle):
        return None
    for x in tokens:
        proxy.append([x.value, orederMapping.get(x.curType, orederMapping['<UNK>'])])

    if len(proxy) == 0:
        return None

    # 6. Split subtokens and filter (same as dataextractor.py)
    result = []
    for item in proxy:
        subtokens = get_subtokens(item[0])
        for subtoken in subtokens:
            if is_valid_subtoken(subtoken):
                result.append([subtoken, item[1]])

    if len(result) < 3:
        return None

    # Append the oracle last
    result.append(oracle)
    return result


def convert_dataset(input_path: str, output_path: str, with_context: bool = True):
    """Convert a whole dataset file."""
    ctx_label = "with_context" if with_context else "without_context"
    print(f"Reading: {input_path} [{ctx_label}]")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Samples: {len(data)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converted = 0
    skipped = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(data):
            result = convert_sample(sample, with_context=with_context)
            if result is not None:
                f.write(json.dumps(result, separators=(',', ':'), ensure_ascii=False) + '\n')
                converted += 1
            else:
                skipped += 1

            if (i + 1) % 5000 == 0:
                print(f"  Progress: {i+1}/{len(data)}, converted: {converted}, skipped: {skipped}")

    print(f"Output: {output_path}")
    print(f"Converted: {converted}, skipped: {skipped}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert the new datasets into Cognac input format (using javalang)")
    parser.add_argument('--type', choices=['random', 'mask_recursive', 'original', 'original_ctx', 'all'],
                        help='dataset type to convert')
    parser.add_argument('--context', choices=['with', 'without', 'all'],
                        help='context mode: with=with callee, without=without callee, all=generate both')
    parser.add_argument('--generate-all', action='store_true',
                        help='generate the data for experiments 1-9 (3 datasets x context x train/test)')
    parser.add_argument('--input-dir', default=None,
                        help='input dataset directory (default: ./raw_data/)')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: ./dataset/)')
    args = parser.parse_args()

    if args.generate_all:
        args.type = 'all'
        args.context = 'all'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(script_dir, 'raw_data')
    output_dir = args.output_dir or os.path.join(script_dir, 'dataset')

    datasets = []
    if args.type in ('random', 'all'):
        datasets.append(('random', 'train_d_2_b_128_l_16_random_no_test.json',
                         'test_d_2_b_128_l_16_random_no_test.json'))
    if args.type in ('mask_recursive', 'all'):
        datasets.append(('mask_recursive', 'train_d_2_b_128_l_16_mask_recursive_no_test.json',
                         'test_d_2_b_128_l_16_mask_recursive_no_test.json'))
    if args.type in ('original', 'all'):
        datasets.append(('original', 'train_d_1_b_128_l_16_no_test.json',
                         'test_d_1_b_128_l_16_no_test.json'))
    # original_ctx: original data with visible names (d_1) + completed callee-body context (children filled with real names)
    # Used only for the with_context group; excluded from --generate-all (to avoid confusion with original, which has empty children)
    if args.type == 'original_ctx':
        datasets.append(('original_ctx', 'train_d_1_with_callee_body.json',
                         'test_d_1_with_callee_body.json'))

    context_modes = []
    if args.context in ('with', 'all'):
        context_modes.append(('with_context', True))
    if args.context in ('without', 'all'):
        context_modes.append(('without_context', False))

    for ds_name, train_file, test_file in datasets:
        for ctx_name, with_ctx in context_modes:
            print(f"=== Converting {ds_name} / {ctx_name} ===")
            ds_output_dir = os.path.join(output_dir, ds_name, ctx_name)

            train_input = os.path.join(input_dir, train_file)
            test_input = os.path.join(input_dir, test_file)

            if not os.path.exists(train_input):
                print(f"Warning: {train_input} not found, skipping")
                continue

            convert_dataset(train_input, os.path.join(ds_output_dir, 'train.json'), with_context=with_ctx)

            if os.path.exists(test_input):
                convert_dataset(test_input, os.path.join(ds_output_dir, 'test.json'), with_context=with_ctx)
            else:
                print(f"Warning: {test_input} not found, skipping test set")


if __name__ == '__main__':
    main()
