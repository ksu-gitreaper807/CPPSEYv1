#!/usr/bin/env python3
"""Repeat the synthetic evaluation matrix and save raw measurements as JSON."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from run_eval import ALARM_RE, BIN, CASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--all', action='store_true', help='measure the full 33-case matrix')
    parser.add_argument('--output', type=Path, default=ROOT / 'test/results/benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not BIN.exists():
        parser.error('build the harness with make -C test first')

    cpu = next((line.split(':', 1)[1].strip() for line in
                Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    report = {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'cpu': cpu, 'platform': platform.platform(),
        'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        'source_base_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'binary_sha256': hashlib.sha256(BIN.read_bytes()).hexdigest(),
        'method': 'Sequential synthetic harness subprocesses; diagnostics and protection disabled. '
                  'One unmeasured warm-up per case, then repeated wall-clock measurements including '
                  'startup, procedural pixels, conversion, fake downsampling, analysis and alarm output. '
                  'Source-time alarm timestamps are not wall-clock latency. No live capture measured.',
        'repeats': args.repeats, 'cases': [],
    }
    failed = False
    selected = CASES if args.all else [entry for entry in CASES if entry[7] in (
        'full-screen 3.0 Hz', 'full-screen 6.0 Hz',
        '6.0 Hz @ 60fps, 1080p native', 'static grey')]
    for case, fps, hz, l1, l2, frames, size, label, expectation in selected:
        env = {key: value for key, value in os.environ.items() if not key.startswith('PSE_')}
        env.update(PSE_SYNTH_CASE=case, PSE_SYNTH_FPS=str(fps), PSE_SYNTH_HZ=str(hz),
                   PSE_SYNTH_L1=str(l1), PSE_SYNTH_L2=str(l2),
                   PSE_SYNTH_FRAMES=str(frames), PSE_SYNTH_SIZE=size,
                   PSE_NO_PROTECTION='1', PSE_DIAGNOSTICS='0', PSE_MAX_FRAMES='0')
        samples = []
        for repeat in range(args.repeats + 1):
            start = time.perf_counter()
            result = subprocess.run([str(BIN), 'desktop', str(fps)], env=env,
                                    capture_output=True, text=True, timeout=120)
            elapsed = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(f'{label}: exit {result.returncode}: {result.stderr[-2000:]}')
            captured = re.search(r'Captured (\d+) frames', result.stdout)
            if not captured or int(captured[1]) != frames:
                raise RuntimeError(f'{label}: missing or incorrect captured-frame count')
            alarms = ALARM_RE.findall(result.stdout)
            passed = (bool(alarms) if expectation == 'pos' else
                      not alarms if expectation == 'neg' else True)
            failed |= not passed
            if repeat:
                samples.append({'wall_seconds': elapsed, 'throughput_fps': frames / elapsed,
                                'alarms': len(alarms), 'passed': passed,
                                'first_alarm_source_seconds': float(alarms[0]) if alarms else None})
        median = statistics.median(sample['wall_seconds'] for sample in samples)
        row = {'label': label, 'case': case, 'source_fps': fps, 'hz': hz,
               'frames': frames, 'size': size, 'expectation': expectation,
               'l1': l1, 'l2': l2, 'samples': samples,
               'median_wall_seconds': median, 'median_throughput_fps': frames / median,
               'passed': all(sample['passed'] for sample in samples)}
        report['cases'].append(row)
        print(f"{label:44} {frames / median:8.1f} fps  "
              f"{median * 1000 / frames:6.3f} ms/frame  "
              f"{'PASS' if row['passed'] else 'FAIL'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Raw measurements: {args.output}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
