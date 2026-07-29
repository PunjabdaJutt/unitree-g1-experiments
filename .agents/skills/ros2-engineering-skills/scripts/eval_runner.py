#!/usr/bin/env python3
"""Skills 2.0 Eval Runner — Structural smoke check for skill evals.

Loads eval definitions from eval.yaml, validates that prompt/expected files
exist, and reports whether each expected file's text addresses the keywords
declared in its criteria. This is a *structural* check: it does not invoke
Claude, does not execute the skill, and does not measure model accuracy.

A criterion "passes" when at least 30% of its non-stop-word tokens appear
(case-insensitively) in the expected file. The threshold and matching are
deliberately permissive — the goal is to catch missing or empty fixtures,
not to score model output. Treat the pass rate as a fixture-quality
indicator, not a benchmark.

Usage:
    python eval_runner.py [--eval-dir DIR] [--eval-name NAME] [--json] [--verbose]

Exit codes:
    0 — All evals passed
    1 — One or more evals failed
    2 — Configuration error
"""

__version__ = '1.0.0'

import argparse
import json
import os
import re
import sys
import time


def _import_yaml():
    """Import PyYAML lazily, only on the execution path that needs it.

    Importing (or asking for --help from) this module must not require
    PyYAML: the module is imported by the test suite and the CLI should
    print usage on a bare environment. Only actually running evals needs
    the dependency, so the friendly failure lives here, not at module
    import time.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print('Error: PyYAML is required to run evals — install the '
              'development dependencies first: '
              'pip install -r requirements-dev.txt', file=sys.stderr)
        sys.exit(2)
    return yaml


def load_eval_config(eval_dir):
    """Load and validate eval.yaml from the given directory.

    Returns:
        dict: Parsed eval configuration.

    Raises:
        SystemExit: If eval.yaml is missing or malformed, or PyYAML is
            not installed.
    """
    yaml = _import_yaml()
    config_path = os.path.join(eval_dir, 'eval.yaml')
    if not os.path.isfile(config_path):
        print(f'Error: eval.yaml not found at {config_path}', file=sys.stderr)
        sys.exit(2)

    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            config = yaml.safe_load(fh)
    except yaml.YAMLError as e:
        print(f'Error: failed to parse eval.yaml: {e}', file=sys.stderr)
        sys.exit(2)

    if not isinstance(config, dict):
        print('Error: eval.yaml must be a YAML mapping', file=sys.stderr)
        sys.exit(2)

    if 'evals' not in config:
        print('Error: eval.yaml must contain an "evals" key', file=sys.stderr)
        sys.exit(2)

    if not isinstance(config['evals'], list):
        print('Error: "evals" must be a list', file=sys.stderr)
        sys.exit(2)

    return config


def _resolve_within(base_dir, rel_path):
    """Resolve ``rel_path`` against ``base_dir`` and reject paths that escape it.

    Returns the resolved absolute path on success, or ``None`` if the path
    escapes ``base_dir`` (e.g. ``../../etc/passwd``).
    """
    base = os.path.realpath(base_dir)
    candidate = os.path.realpath(os.path.join(base, rel_path))
    try:
        if os.path.commonpath([base, candidate]) != base:
            return None
    except ValueError:
        # Different drives on Windows, or other commonpath edge cases.
        return None
    return candidate


def validate_eval_entry(entry, eval_dir):
    """Validate a single eval entry has all required fields and files exist.

    Returns:
        list[str]: List of validation error messages (empty = valid).
    """
    errors = []
    required_fields = ['name', 'prompt', 'expected', 'criteria']
    for field in required_fields:
        if field not in entry:
            errors.append(f'Missing required field: {field}')

    # Resolve and cache paths on the entry so run_eval can reuse them
    # without re-resolving (avoids a TOCTOU window between validation and
    # use, and removes a duplicate commonpath call).
    if 'prompt' in entry:
        prompt_path = _resolve_within(eval_dir, entry['prompt'])
        if prompt_path is None:
            errors.append(f'Prompt path escapes eval dir: {entry["prompt"]}')
        elif not os.path.isfile(prompt_path):
            errors.append(f'Prompt file not found: {prompt_path}')
        else:
            entry['_resolved_prompt'] = prompt_path

    if 'expected' in entry:
        expected_path = _resolve_within(eval_dir, entry['expected'])
        if expected_path is None:
            errors.append(
                f'Expected path escapes eval dir: {entry["expected"]}')
        elif not os.path.isfile(expected_path):
            errors.append(f'Expected file not found: {expected_path}')
        else:
            entry['_resolved_expected'] = expected_path

    if 'criteria' in entry:
        if not isinstance(entry['criteria'], list):
            errors.append('"criteria" must be a list')
        else:
            for i, criterion in enumerate(entry['criteria']):
                if isinstance(criterion, dict):
                    if 'description' not in criterion:
                        errors.append(
                            f'Criterion {i} missing "description"')
                elif not isinstance(criterion, str):
                    errors.append(
                        f'Criterion {i} must be a string or dict')

    if 'timeout' in entry:
        if not isinstance(entry['timeout'], (int, float)):
            errors.append('"timeout" must be a number')
        elif entry['timeout'] <= 0:
            errors.append('"timeout" must be positive')

    return errors


def load_file_content(filepath):
    """Load content from a file, returning empty string on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        return ''


def extract_criteria_text(criteria):
    """Extract text descriptions from criteria list (supports str and dict)."""
    texts = []
    for c in criteria:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, dict):
            texts.append(c.get('description', str(c)))
    return texts


def _term_matches(term, expected_lower):
    """Return True if *term* matches any word in *expected_lower*.

    Matching is morphology-tolerant: a 4+ char term matches any word whose
    word boundary shares the first 4 chars AND where either side is a prefix
    of the other. This handles common inflections without a full stemmer:

        criterion 'paths'    -> matches 'path' / 'paths' / 'pathway' (no — 'paths' is not a prefix of 'pathway')
        criterion 'warnings' -> matches 'warn' / 'warning' / 'warnings' / 'warned'
        criterion 'service'  -> matches 'services' / 'serviced'
        criterion 'process'  -> 'proces' false-positive avoided (no 'proces' in real text)

    Words under 4 chars fall back to exact word-boundary matching so short
    common tokens (e.g. 'qos', 'tf') do not over-match.
    """
    if len(term) < 4:
        return re.search(r'\b' + re.escape(term) + r'\b',
                         expected_lower) is not None
    # 4-char prefix anchor: cheap pre-filter so we only run the prefix test
    # on candidate words, not on every word in the file.
    prefix = re.escape(term[:4])
    pattern = r'\b' + prefix + r'\w*\b'
    for match in re.finditer(pattern, expected_lower):
        word = match.group(0)
        # Bidirectional prefix check: morphological variant must share a stem.
        # 'paths' / 'path': 'path' (4) is prefix of both -> the shorter side
        # is always a prefix of the longer. Reject if neither is a prefix
        # of the other (e.g. 'paths' vs 'patient' — share 'pat' only).
        if word.startswith(term) or term.startswith(word):
            return True
    return False


def evaluate_criteria(expected_content, criteria_texts,
                      coverage_threshold=0.30, min_terms=3):
    """Evaluate criteria against expected content.

    This is a structural validation — it checks that the expected file
    contains content that addresses each criterion via morphology-tolerant
    keyword matching (see _term_matches).

    Args:
        expected_content: Text of the expected file.
        criteria_texts: List of criterion description strings.
        coverage_threshold: Fraction of key terms that must match for the
            criterion to pass. Default 0.30 (a permissive structural check,
            not a benchmark). Configurable via --min-coverage CLI flag.
        min_terms: Minimum number of meaningful key terms a criterion must
            have to be evaluated; shorter criteria are not informative enough
            to score and fail by default. Default 3.

    Returns:
        list[dict]: Results for each criterion with pass/fail and details.
    """
    results = []
    for criterion_text in criteria_texts:
        # Extract key terms from the criterion for matching
        key_terms = []
        words = criterion_text.lower().split()
        # Filter out common words to find meaningful terms
        stop_words = {
            'must', 'should', 'the', 'a', 'an', 'is', 'are', 'for',
            'and', 'or', 'of', 'in', 'to', 'with', 'that', 'this',
            'be', 'have', 'has', 'not', 'from', 'at', 'by', 'on',
        }
        for word in words:
            cleaned = word.strip('"\'.,;:!?()[]{}')
            if cleaned and cleaned not in stop_words and len(cleaned) > 2:
                key_terms.append(cleaned)

        expected_lower = expected_content.lower()
        matched_terms = [t for t in key_terms
                         if _term_matches(t, expected_lower)]
        coverage = len(matched_terms) / max(len(key_terms), 1)
        passed = (len(key_terms) >= min_terms
                  and coverage >= coverage_threshold)

        results.append({
            'criterion': criterion_text,
            'passed': passed,
            'coverage': round(coverage, 2),
            'matched_terms': matched_terms,
            'total_terms': len(key_terms),
        })

    return results


def _extract_criteria_with_weights(criteria_entries):
    """Return list of (text, weight) tuples for criteria.

    Supports both string and dict forms. Weight defaults to 1.0 when omitted
    (so old eval.yaml entries without weight behave as before).
    """
    pairs = []
    for c in criteria_entries:
        if isinstance(c, str):
            pairs.append((c, 1.0))
        elif isinstance(c, dict):
            text = c.get('description', str(c))
            weight = c.get('weight', 1.0)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            if weight < 0:
                weight = 0.0
            pairs.append((text, weight))
    return pairs


def _content_path_for_source(eval_dir, eval_name, source):
    """Return the file path that backs a given content `source` for an eval.

    Sources:
        'expected' - the reference/ideal answer (fixture).
        'output'   - the actual model output captured by the user with the
                     skill loaded (evals/outputs/{name}.md).
        'baseline' - the actual model output captured WITHOUT the skill loaded
                     (evals/outputs_baseline/{name}.md), used for parity.
    """
    if source == 'expected':
        return None  # caller uses entry['_resolved_expected']
    sub = {'output': 'outputs',
           'baseline': 'outputs_baseline'}[source]
    return os.path.join(eval_dir, sub, f'{eval_name}.md')


def run_eval(entry, eval_dir, verbose=False,
             coverage_threshold=0.30, pass_rate_threshold=80.0,
             content_source='expected'):
    """Run a single eval and return results.

    Args:
        entry: eval definition (dict from eval.yaml).
        eval_dir: directory containing eval.yaml.
        verbose: include prompt/expected paths and lengths in result.
        coverage_threshold: per-criterion key-term coverage required to pass.
        pass_rate_threshold: overall weighted pass rate for the eval to be
            considered passing. Both thresholds are CLI-configurable so the
            same code can be used both for a permissive structural smoke
            check and for stricter local validation.
        content_source: 'expected' (default, current fixture behaviour),
            'output' (real model output captured under outputs/), or
            'baseline' (output captured without the skill loaded). Returns
            status='skipped' if the source file is missing — this lets
            parity_test report "not yet captured" without failing CI.

    Returns:
        dict: Eval results including pass/fail, timing, and criteria breakdown.
    """
    start_time = time.monotonic()

    # Validate entry
    validation_errors = validate_eval_entry(entry, eval_dir)
    if validation_errors:
        elapsed = (time.monotonic() - start_time) * 1000
        return {
            'name': entry.get('name', '<unknown>'),
            'status': 'error',
            'errors': validation_errors,
            'execution_time_ms': round(elapsed, 1),
            'criteria_results': [],
            'pass_rate': 0.0,
        }

    # Reuse the paths resolved during validation (no re-resolution, which
    # avoids any TOCTOU window between validate_eval_entry and here).
    prompt_path = entry['_resolved_prompt']

    # Pick which file backs the criteria check. Default 'expected' uses the
    # fixture; 'output'/'baseline' look at user-captured model outputs.
    if content_source == 'expected':
        content_path = entry['_resolved_expected']
    else:
        content_path = _content_path_for_source(
            eval_dir, entry['name'], content_source)
        if not os.path.isfile(content_path):
            elapsed = (time.monotonic() - start_time) * 1000
            return {
                'name': entry['name'],
                'status': 'skipped',
                'reason': (f'No {content_source} captured yet: '
                           f'expected file at {content_path}. See '
                           'docs/EVAL_WORKFLOW.md for how to populate.'),
                'execution_time_ms': round(elapsed, 1),
                'criteria_results': [],
                'pass_rate': 0.0,
                'content_source': content_source,
            }

    prompt_content = load_file_content(prompt_path)
    expected_content = load_file_content(content_path)

    if not prompt_content:
        elapsed = (time.monotonic() - start_time) * 1000
        return {
            'name': entry['name'],
            'status': 'error',
            'errors': [f'Empty prompt file: {prompt_path}'],
            'execution_time_ms': round(elapsed, 1),
            'criteria_results': [],
            'pass_rate': 0.0,
        }

    if not expected_content:
        elapsed = (time.monotonic() - start_time) * 1000
        return {
            'name': entry['name'],
            'status': 'error',
            'errors': [f'Empty {content_source} file: {content_path}'],
            'execution_time_ms': round(elapsed, 1),
            'criteria_results': [],
            'pass_rate': 0.0,
        }

    # Extract criteria + weights, evaluate, then compute weighted pass rate.
    criteria_pairs = _extract_criteria_with_weights(entry['criteria'])
    criteria_texts = [t for t, _ in criteria_pairs]
    weights = [w for _, w in criteria_pairs]
    criteria_results = evaluate_criteria(
        expected_content, criteria_texts,
        coverage_threshold=coverage_threshold)
    # Attach weight to each result (downstream reports can show it).
    for r, w in zip(criteria_results, weights):
        r['weight'] = w

    passed_count = sum(1 for r in criteria_results if r['passed'])
    total_count = len(criteria_results)
    weighted_passed = sum(w for r, w in zip(criteria_results, weights)
                          if r['passed'])
    weighted_total = sum(weights)
    # Use weighted pass rate when weights are non-uniform; fall back to
    # simple ratio if all weights are 0 or the eval has no criteria.
    if weighted_total > 0:
        pass_rate = (weighted_passed / weighted_total) * 100
    else:
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0

    elapsed = (time.monotonic() - start_time) * 1000

    # Estimate token count (rough: ~4 chars per token)
    token_estimate = (len(prompt_content) + len(expected_content)) // 4

    status = 'pass' if pass_rate >= pass_rate_threshold else 'fail'

    result = {
        'name': entry['name'],
        'status': status,
        'pass_rate': round(pass_rate, 1),
        'passed_criteria': passed_count,
        'total_criteria': total_count,
        'weighted_passed': round(weighted_passed, 2),
        'weighted_total': round(weighted_total, 2),
        'execution_time_ms': round(elapsed, 1),
        'token_estimate': token_estimate,
        'criteria_results': criteria_results,
        'tags': entry.get('tags', []),
    }

    if verbose:
        result['prompt_path'] = prompt_path
        # `expected_path` kept as the field name for backwards compat with
        # tooling that consumed previous verbose output; the value now
        # reflects whichever content_source was scored.
        result['expected_path'] = content_path
        result['content_source'] = content_source
        result['prompt_length'] = len(prompt_content)
        result['expected_length'] = len(expected_content)

    return result


def run_all_evals(config, eval_dir, eval_name=None, verbose=False,
                  coverage_threshold=0.30, pass_rate_threshold=80.0,
                  content_source='expected'):
    """Run all evals (or a specific one) and return aggregate results.

    Returns:
        dict: Aggregate results including per-eval and summary data.
    """
    evals = config['evals']
    if eval_name:
        evals = [e for e in evals if e.get('name') == eval_name]
        if not evals:
            print(f'Error: eval "{eval_name}" not found', file=sys.stderr)
            sys.exit(2)

    results = []
    for entry in evals:
        result = run_eval(
            entry, eval_dir, verbose=verbose,
            coverage_threshold=coverage_threshold,
            pass_rate_threshold=pass_rate_threshold,
            content_source=content_source)
        results.append(result)

    total_evals = len(results)
    passed_evals = sum(1 for r in results if r['status'] == 'pass')
    failed_evals = sum(1 for r in results if r['status'] == 'fail')
    error_evals = sum(1 for r in results if r['status'] == 'error')
    skipped_evals = sum(1 for r in results if r['status'] == 'skipped')
    scored_count = passed_evals + failed_evals
    avg_pass_rate = (
        sum(r['pass_rate'] for r in results
            if r['status'] in ('pass', 'fail')) / scored_count
        if scored_count > 0 else 0.0
    )
    total_time = sum(r['execution_time_ms'] for r in results)

    # Overall: 'no_data' if everything skipped (no model outputs captured),
    # 'fail' if any scored eval failed or errored, otherwise 'pass'. The
    # 'no_data' state matters for judge mode in CI - we do not want a clean
    # green when the user simply has not pasted anything yet.
    if failed_evals == 0 and error_evals == 0:
        overall = 'no_data' if (
            scored_count == 0 and skipped_evals > 0) else 'pass'
    else:
        overall = 'fail'

    return {
        'skill': config.get('skill', '<unknown>'),
        'version': config.get('version', '<unknown>'),
        'classification': config.get('classification', '<unknown>'),
        'deprecation_risk': config.get('deprecation-risk', '<unknown>'),
        'content_source': content_source,
        'summary': {
            'total_evals': total_evals,
            'passed': passed_evals,
            'failed': failed_evals,
            'errors': error_evals,
            'skipped': skipped_evals,
            'average_pass_rate': round(avg_pass_rate, 1),
            'total_execution_time_ms': round(total_time, 1),
            'overall_status': overall,
        },
        'evals': results,
        'parity_test': config.get('parity_test', None),
    }


def run_parity_test(config, eval_dir, verbose=False,
                    coverage_threshold=0.30, pass_rate_threshold=80.0):
    """Score the skill ON vs OFF and report delta + deprecation status.

    For each eval:
      * Score evals/outputs/{name}.md            -> skill_on_pass_rate
      * Score evals/outputs_baseline/{name}.md   -> skill_off_pass_rate
      * delta = on - off

    Evals missing either capture are marked skipped, do not break the run,
    but do exclude themselves from the delta aggregate.

    Aggregate delta is compared against eval.yaml's
    `parity_test.threshold` (default 5.0%). The result is appended to
    `evals/history/<UTC ISO date>.json` as JSON-lines. If the most recent
    `parity_test.consecutive_failures_for_deprecation` runs all sit under
    threshold, the report flags the skill as a deprecation candidate.
    """
    parity_cfg = config.get('parity_test') or {}
    threshold = float(parity_cfg.get('threshold', 5.0))
    consec = int(
        parity_cfg.get('consecutive_failures_for_deprecation', 3))

    on_report = run_all_evals(
        config, eval_dir, verbose=verbose,
        coverage_threshold=coverage_threshold,
        pass_rate_threshold=pass_rate_threshold,
        content_source='output')
    off_report = run_all_evals(
        config, eval_dir, verbose=verbose,
        coverage_threshold=coverage_threshold,
        pass_rate_threshold=pass_rate_threshold,
        content_source='baseline')

    off_by_name = {ev['name']: ev for ev in off_report['evals']}
    deltas = []
    per_eval = []
    for ev_on in on_report['evals']:
        ev_off = off_by_name.get(ev_on['name'], {})
        if (ev_on.get('status') == 'skipped'
                or ev_off.get('status') == 'skipped'):
            per_eval.append({
                'name': ev_on['name'],
                'status': 'skipped',
                'reason': (ev_on.get('reason')
                           or ev_off.get('reason')
                           or 'missing capture on one side'),
            })
            continue
        on_rate = ev_on.get('pass_rate', 0.0)
        off_rate = ev_off.get('pass_rate', 0.0)
        delta = on_rate - off_rate
        deltas.append(delta)
        per_eval.append({
            'name': ev_on['name'],
            'status': 'scored',
            'skill_on_pass_rate': on_rate,
            'skill_off_pass_rate': off_rate,
            'delta': round(delta, 2),
            'meets_threshold': delta >= threshold,
        })

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    threshold_met = bool(deltas) and avg_delta >= threshold

    # Append to history (JSON-lines).
    history_dir = os.path.join(eval_dir, 'history')
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(
        history_dir,
        time.strftime('%Y-%m', time.gmtime()) + '.jsonl')
    history_entry = {
        'timestamp_utc': time.strftime(
            '%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'avg_delta': round(avg_delta, 2),
        'threshold': threshold,
        'threshold_met': threshold_met,
        'scored_evals': len(deltas),
        'skipped_evals': len(per_eval) - len(deltas),
        'per_eval': per_eval,
    }
    try:
        with open(history_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(history_entry) + '\n')
    except OSError:
        pass  # history is best-effort, do not crash the run

    deprecation_candidate = _check_deprecation_streak(
        history_dir, threshold, consec)

    return {
        'skill': config.get('skill', '<unknown>'),
        'version': config.get('version', '<unknown>'),
        'mode': 'parity',
        'threshold': threshold,
        'consecutive_failures_for_deprecation': consec,
        'avg_delta': round(avg_delta, 2),
        'threshold_met': threshold_met,
        'deprecation_candidate': deprecation_candidate,
        'scored_evals': len(deltas),
        'skipped_evals': len(per_eval) - len(deltas),
        'per_eval': per_eval,
        'history_file': history_path,
    }


def _check_deprecation_streak(history_dir, threshold, consec_required):
    """Return True if the most recent `consec_required` history entries all
    failed to meet `threshold`. Returns False if not enough history exists
    yet (need at least `consec_required` runs to declare deprecation).
    """
    if consec_required <= 0:
        return False
    try:
        files = sorted(
            (f for f in os.listdir(history_dir) if f.endswith('.jsonl')),
            reverse=True)
    except OSError:
        return False
    entries = []
    for fname in files:
        try:
            with open(os.path.join(history_dir, fname),
                      'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    # Sort by timestamp descending and take the most recent N.
    entries.sort(key=lambda e: e.get('timestamp_utc', ''), reverse=True)
    recent = entries[:consec_required]
    if len(recent) < consec_required:
        return False
    return all(not e.get('threshold_met', False) for e in recent)


def print_report(report):
    """Print a human-readable eval report."""
    print('=' * 70)
    print(f'  Skills 2.0 Eval Report: {report["skill"]} v{report["version"]}')
    print(f'  Classification: {report["classification"]}  |  '
          f'Deprecation Risk: {report["deprecation_risk"]}')
    print('=' * 70)
    print()

    summary = report['summary']
    skipped = summary.get('skipped', 0)
    scored = summary['passed'] + summary['failed']
    # If everything was skipped (no model outputs captured yet) treat the
    # whole run as 'no data' rather than spuriously claiming PASS.
    if scored == 0 and skipped > 0:
        status_icon = 'NODATA'
    else:
        status_icon = 'PASS' if summary['overall_status'] == 'pass' else 'FAIL'
    parts = [
        f'  Overall: [{status_icon}]',
        f'{summary["passed"]}/{summary["total_evals"]} passed',
        f'({summary["average_pass_rate"]}% avg)',
        f'{summary["total_execution_time_ms"]:.0f}ms total',
    ]
    if skipped:
        parts.insert(2, f'skipped: {skipped}')
    print('  '.join(parts))
    source = report.get('content_source', 'expected')
    if source != 'expected':
        print(f'  Content source: {source} '
              f'(use --mode=structural for the fixture-only smoke check)')
    print()

    for ev in report['evals']:
        if ev['status'] == 'skipped':
            print(f'  [SKIP] {ev["name"]}')
            reason = ev.get('reason', 'no content available')
            print(f'         {reason}')
            print()
            continue
        icon = 'PASS' if ev['status'] == 'pass' else (
            'FAIL' if ev['status'] == 'fail' else 'ERR ')
        print(f'  [{icon}] {ev["name"]}')
        print(f'         Pass rate: {ev["pass_rate"]}%  '
              f'({ev.get("passed_criteria", 0)}/{ev.get("total_criteria", 0)} criteria)  '
              f'{ev["execution_time_ms"]:.1f}ms')

        if ev.get('errors'):
            for err in ev['errors']:
                print(f'         ERROR: {err}')

        if ev.get('criteria_results'):
            for cr in ev['criteria_results']:
                cr_icon = '+' if cr['passed'] else '-'
                print(f'         [{cr_icon}] {cr["criterion"][:60]}...'
                      if len(cr['criterion']) > 60
                      else f'         [{cr_icon}] {cr["criterion"]}')
        print()

    if report.get('parity_test') and report['parity_test'].get('enabled'):
        pt = report['parity_test']
        print('-' * 70)
        print(f'  Parity Test: enabled (threshold: {pt["threshold"]}%)')
        print(f'  Consecutive failures for deprecation: '
              f'{pt["consecutive_failures_for_deprecation"]}')
        print()

    print('=' * 70)


def _print_parity_report(report):
    """Human-readable parity report (skill ON vs OFF)."""
    print('=' * 70)
    print(f'  Skills 2.0 Parity Test: {report["skill"]} v{report["version"]}')
    print(f'  Threshold: avg_delta >= {report["threshold"]}%  |  '
          f'Deprecation streak: {report["consecutive_failures_for_deprecation"]}')
    print('=' * 70)
    print()
    threshold_icon = 'MET' if report['threshold_met'] else 'MISS'
    print(f'  Average delta:        {report["avg_delta"]:+.2f}%  '
          f'[{threshold_icon}]')
    print(f'  Scored evals:         {report["scored_evals"]}  '
          f'(skipped: {report["skipped_evals"]})')
    if report['deprecation_candidate']:
        print(f'  *** DEPRECATION CANDIDATE: most recent '
              f'{report["consecutive_failures_for_deprecation"]} runs all '
              f'below threshold ***')
    print(f'  History file:         {report["history_file"]}')
    print()
    print('-' * 70)
    for ev in report['per_eval']:
        if ev['status'] == 'skipped':
            print(f'  [SKIP] {ev["name"]:40s}  {ev["reason"]}')
        else:
            icon = '+' if ev['meets_threshold'] else '-'
            print(f'  [{icon}]    {ev["name"]:40s}  '
                  f'on={ev["skill_on_pass_rate"]:5.1f}%  '
                  f'off={ev["skill_off_pass_rate"]:5.1f}%  '
                  f'delta={ev["delta"]:+.1f}%')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Skills 2.0 Eval Runner - Automated quality verification')
    parser.add_argument(
        '--eval-dir', default=None,
        help='Directory containing eval.yaml (default: evals/ relative to skill root)')
    parser.add_argument(
        '--eval-name', default=None,
        help='Run a specific eval by name')
    parser.add_argument(
        '--json', action='store_true', dest='json_output',
        help='Output results as JSON')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Include additional details in output')
    parser.add_argument(
        '--min-coverage', type=float, default=0.30,
        metavar='FRAC',
        help=('Per-criterion key-term coverage required to pass '
              '(default: 0.30 - permissive structural smoke check)'))
    parser.add_argument(
        '--min-pass-rate', type=float, default=80.0,
        metavar='PCT',
        help=('Overall weighted pass rate (0-100) required for an eval to '
              'be considered passing (default: 80.0)'))
    parser.add_argument(
        '--mode',
        choices=['structural', 'judge'],
        default=None,  # resolved to 'structural' below; None distinguishes
                       # an explicit --mode from the default for the
                       # --parity exclusivity check
        help=('structural (default): score evals/expected fixtures - cheap '
              'CI gate. judge: score user-pasted real model outputs under '
              'evals/outputs/ - see docs/EVAL_WORKFLOW.md'))
    parser.add_argument(
        '--parity', action='store_true', default=False,
        help=('Run parity test: score skill ON (evals/outputs/) vs OFF '
              '(evals/outputs_baseline/), append delta to evals/history/, '
              'and flag deprecation candidacy if recent runs miss threshold. '
              'Mutually exclusive with --mode.'))
    parser.add_argument(
        '--version', action='version',
        version=f'%(prog)s {__version__}')

    args = parser.parse_args()
    if not 0.0 <= args.min_coverage <= 1.0:
        print(
            f'Error: --min-coverage must be in [0.0, 1.0], '
            f'got: {args.min_coverage}', file=sys.stderr)
        sys.exit(2)
    if not 0.0 <= args.min_pass_rate <= 100.0:
        print(
            f'Error: --min-pass-rate must be in [0.0, 100.0], '
            f'got: {args.min_pass_rate}', file=sys.stderr)
        sys.exit(2)

    # Determine eval directory
    if args.eval_dir:
        eval_dir = args.eval_dir
    else:
        skill_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_dir = os.path.join(skill_root, 'evals')

    config = load_eval_config(eval_dir)

    if args.parity:
        if args.mode is not None:
            print('Error: --parity and --mode are mutually exclusive; '
                  'parity is its own scoring mode.', file=sys.stderr)
            sys.exit(2)
        if args.eval_name:
            print('Error: --parity does not support --eval-name; runs all '
                  'evals to compute aggregate delta.', file=sys.stderr)
            sys.exit(2)
        report = run_parity_test(
            config, eval_dir, verbose=args.verbose,
            coverage_threshold=args.min_coverage,
            pass_rate_threshold=args.min_pass_rate)
        if args.json_output:
            print(json.dumps(report, indent=2))
        else:
            _print_parity_report(report)
        # Exit non-zero ONLY for deprecation candidacy; missing threshold
        # on a single run is informational, not a CI failure.
        sys.exit(2 if report['deprecation_candidate'] else 0)

    mode = args.mode or 'structural'
    content_source = 'output' if mode == 'judge' else 'expected'
    report = run_all_evals(config, eval_dir,
                           eval_name=args.eval_name,
                           verbose=args.verbose,
                           coverage_threshold=args.min_coverage,
                           pass_rate_threshold=args.min_pass_rate,
                           content_source=content_source)

    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    status = report['summary']['overall_status']
    # 'no_data' (judge mode, nothing captured yet) is not a CI failure -
    # exit 0 so users can wire judge mode into CI without it failing until
    # outputs are populated. Only an explicit fail/error returns non-zero.
    sys.exit(1 if status == 'fail' else 0)


if __name__ == '__main__':
    main()
