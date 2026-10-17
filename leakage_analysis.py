import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')


def camel_split(name):
    """Split camelCase/PascalCase into sub-tokens (lowercase)."""
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    tokens = re.sub(r'([a-z])([A-Z])', r'\1 \2', tokens)
    tokens = tokens.split()
    return [t.lower() for t in tokens if t]


def extract_method_name(full_name):
    """Extract method name from 'pkg.Class.methodName(params)'."""
    name = full_name.split('(')[0]
    parts = name.split('.')
    return parts[-1] if parts else name


def body_contains_token(body, token):
    """Check if token appears in body as a word (case-insensitive)."""
    # Use word boundary matching to avoid partial matches
    pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(token) + r'(?![a-zA-Z])', re.IGNORECASE)
    return bool(pattern.search(body))


def collect_all_bodies(children):
    """Recursively collect all callee method bodies."""
    bodies = []
    for child in children:
        body = child.get('method_body', '')
        if body:
            bodies.append(body)
        sub = child.get('children', [])
        if sub:
            bodies.extend(collect_all_bodies(sub))
    return bodies


def analyze_leakage(data, label=""):
    samples_with_callees = 0
    samples_any_leakage = 0
    samples_full_leakage = 0
    samples_exclusive_leakage = 0

    total_gt_tokens = 0
    leaked_tokens_count = 0
    exclusive_tokens = 0
    accessible_tokens = 0
    accessible_via_target = 0
    accessible_via_callee = 0

    for item in data:
        root = item['root']
        target_name = extract_method_name(root['method_full_name'])
        target_body = root.get('method_body', '')
        children = root.get('children', [])

        if not children:
            continue

        samples_with_callees += 1
        gt_subtokens = camel_split(target_name)

        if not gt_subtokens:
            continue

        # Collect direct callee sub-tokens
        callee_subtokens = set()
        for child in children:
            cn = extract_method_name(child.get('method_full_name', ''))
            for t in camel_split(cn):
                callee_subtokens.add(t)

        # Collect all callee bodies (recursively) for accessible check
        all_callee_bodies = collect_all_bodies(children)

        sample_leaked = 0
        sample_exclusive = 0

        for gt_token in gt_subtokens:
            total_gt_tokens += 1
            if gt_token in callee_subtokens:
                leaked_tokens_count += 1
                sample_leaked += 1
                # Three mutually exclusive categories:
                in_target = body_contains_token(target_body, gt_token)
                in_callee = any(body_contains_token(b, gt_token) for b in all_callee_bodies)

                if in_target:
                    # Token also appears in target body → even no-context models see it
                    accessible_tokens += 1
                    accessible_via_target += 1
                elif in_callee:
                    # Not in target body, but in callee body → only context models see it
                    accessible_tokens += 1
                    accessible_via_callee += 1
                else:
                    # Not in any body → exclusive to callee name, invisible after masking
                    exclusive_tokens += 1
                    sample_exclusive += 1

        if sample_leaked > 0:
            samples_any_leakage += 1
        if sample_leaked == len(gt_subtokens):
            samples_full_leakage += 1
        if sample_exclusive > 0:
            samples_exclusive_leakage += 1

    print(f"\n{'='*60}")
    print(f"  Leakage Prevalence Analysis - {label}")
    print(f"{'='*60}")
    print(f"Total samples:              {len(data)}")
    print(f"Samples with callees:       {samples_with_callees}")
    print(f"Any leakage (>=1 token):    {samples_any_leakage}/{samples_with_callees} "
          f"({100*samples_any_leakage/samples_with_callees:.1f}%)")
    print(f"Full leakage (all tokens):  {samples_full_leakage}/{samples_with_callees} "
          f"({100*samples_full_leakage/samples_with_callees:.1f}%)")
    print(f"Exclusive leakage (samples):{samples_exclusive_leakage}/{samples_with_callees} "
          f"({100*samples_exclusive_leakage/samples_with_callees:.1f}%)")
    print(f"")
    print(f"Token-level leakage:        {leaked_tokens_count}/{total_gt_tokens} "
          f"({100*leaked_tokens_count/total_gt_tokens:.1f}%)")
    print(f"  In target body (any model sees it):     {accessible_via_target}/{total_gt_tokens} "
          f"({100*accessible_via_target/total_gt_tokens:.1f}%)")
    print(f"  Not in target, but in callee body (context model sees it): {accessible_via_callee}/{total_gt_tokens} "
          f"({100*accessible_via_callee/total_gt_tokens:.1f}%)")
    print(f"  Not in any body (exclusive, invisible after mask): {exclusive_tokens}/{total_gt_tokens} "
          f"({100*exclusive_tokens/total_gt_tokens:.1f}%)")

    return {
        'samples_with_callees': samples_with_callees,
        'any_leakage': samples_any_leakage,
        'full_leakage': samples_full_leakage,
        'exclusive_leakage_samples': samples_exclusive_leakage,
        'total_gt_tokens': total_gt_tokens,
        'leaked_tokens': leaked_tokens_count,
        'exclusive_tokens': exclusive_tokens,
        'accessible_tokens': accessible_tokens,
        'accessible_via_target': accessible_via_target,
        'accessible_via_callee': accessible_via_callee,
    }


if __name__ == '__main__':
    print("Loading train data (mask_recursive_no_test)...")
    with open('train_d_2_b_128_l_16_mask_recursive_no_test.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    print("Loading test data (mask_recursive_no_test)...")
    with open('test_d_2_b_128_l_16_mask_recursive_no_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    train_stats = analyze_leakage(train_data, "Train")
    test_stats = analyze_leakage(test_data, "Test")
    all_stats = analyze_leakage(train_data + test_data, "All (Train+Test)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary Table")
    print(f"{'='*60}")
    t, te, a = train_stats, test_stats, all_stats
    print(f"{'Metric':<32} {'Train':>8} {'Test':>8} {'All':>8}")
    print(f"{'-'*60}")
    print(f"{'Samples with callees':<32} {t['samples_with_callees']:>8} {te['samples_with_callees']:>8} {a['samples_with_callees']:>8}")
    print(f"{'Any leakage':<32} {100*t['any_leakage']/t['samples_with_callees']:>7.1f}% {100*te['any_leakage']/te['samples_with_callees']:>7.1f}% {100*a['any_leakage']/a['samples_with_callees']:>7.1f}%")
    print(f"{'Full leakage':<32} {100*t['full_leakage']/t['samples_with_callees']:>7.1f}% {100*te['full_leakage']/te['samples_with_callees']:>7.1f}% {100*a['full_leakage']/a['samples_with_callees']:>7.1f}%")
    print(f"{'Exclusive leakage (samples)':<32} {100*t['exclusive_leakage_samples']/t['samples_with_callees']:>7.1f}% {100*te['exclusive_leakage_samples']/te['samples_with_callees']:>7.1f}% {100*a['exclusive_leakage_samples']/a['samples_with_callees']:>7.1f}%")
    print(f"{'Token-level leakage':<32} {100*t['leaked_tokens']/t['total_gt_tokens']:>7.1f}% {100*te['leaked_tokens']/te['total_gt_tokens']:>7.1f}% {100*a['leaked_tokens']/a['total_gt_tokens']:>7.1f}%")
    print(f"{'  In target body':<32} {100*t['accessible_via_target']/t['total_gt_tokens']:>7.1f}% {100*te['accessible_via_target']/te['total_gt_tokens']:>7.1f}% {100*a['accessible_via_target']/a['total_gt_tokens']:>7.1f}%")
    print(f"{'  Only in callee body':<32} {100*t['accessible_via_callee']/t['total_gt_tokens']:>7.1f}% {100*te['accessible_via_callee']/te['total_gt_tokens']:>7.1f}% {100*a['accessible_via_callee']/a['total_gt_tokens']:>7.1f}%")
    print(f"{'  Exclusive (no body)':<32} {100*t['exclusive_tokens']/t['total_gt_tokens']:>7.1f}% {100*te['exclusive_tokens']/te['total_gt_tokens']:>7.1f}% {100*a['exclusive_tokens']/a['total_gt_tokens']:>7.1f}%")
