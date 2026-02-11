#!/usr/bin/env python3
"""
Organize flat results files saved as ./results/{policy}-{case}.pth into
subfolders expected by visualization notebooks:
  - ./results/{case}/{policy}-{case}.pth

Some special case like 
- (d): th-exp-ablation
- (e): sjf-ablation
Need also be handled

Usage:
  python scripts/phy/organize_result.py [--results-dir ./results] [--link]

Notes:
  - Only files directly under results-dir are considered as flat files to organize.
  - Existing files in subfolders are left untouched.
	- Casenames are already normalized; this script uses them as-is. Any alias handling
	is done in visualization scripts, not here.
"""

import os
import re
import shutil
from typing import Dict, List, Tuple

RESULTS_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'results')
RESULTS_DIR_DEFAULT = os.path.abspath(RESULTS_DIR_DEFAULT)

# No aliasing or normalization is performed here.


def parse_flat_filename(filename: str) -> Tuple[str, str]:
	"""Parse 'policy-case.pth' -> (policy, case). Raises ValueError if invalid."""
	if not filename.endswith('.pth'):
		raise ValueError('Not a .pth file')
	base = os.path.basename(filename)
	# Split on the FIRST '-' only, since case may contain dashes
	idx = base.find('-')
	if idx <= 0:
		raise ValueError(f'Filename does not match policy-case.pth pattern: {base}')
	policy = base[:idx]
	case = base[idx+1:-4]  # strip .pth
	return policy, case


def ensure_dir(path: str):
	os.makedirs(path, exist_ok=True)


def copy_or_link(src: str, dst: str, use_link: bool = False, dry_run: bool = False):
	"""Copy or symlink src -> dst. When dry_run=True, perform no filesystem changes."""
	if dry_run:
		return
	if use_link:
		# Replace existing symlink/file
		if os.path.lexists(dst):
			os.remove(dst)
		os.symlink(os.path.abspath(src), dst)
	else:
		shutil.copy2(src, dst)



def organize_flat_files(results_dir: str, use_link: bool = False, dry_run: bool = False) -> List[str]:
	"""Move/copy flat files under results_dir into case subfolders. Returns log lines."""
	logs: List[str] = []
	for entry in os.listdir(results_dir):
		src_path = os.path.join(results_dir, entry)
		if not os.path.isfile(src_path):
			continue  # skip directories
		try:
			policy, case = parse_flat_filename(entry)
		except ValueError:
			continue

		# Special-case: skip creating subfolders for lbd* cases
		if case.startswith('lbd') or case.startswith('sjf'):
			logs.append(f"Skip special case: {entry} (no case subfolder created)")
			continue

		# Use casename as-is; no normalization
		dst_dir = os.path.join(results_dir, case)
		ensure_dir(dst_dir)
		dst_name = f"{policy}-{case}.pth"
		dst_path = os.path.join(dst_dir, dst_name)

		copy_or_link(src_path, dst_path, use_link, dry_run=dry_run)
		logs.append(f"Organized: {entry} -> {case}/{dst_name}")
	return logs


# Alias creation removed; visualization handles alias cases.


def build_results_index(results_dir: str) -> Dict[str, Dict[str, str]]:
	"""Build an index: policy -> case -> file_path for organized subfolder files."""
	index: Dict[str, Dict[str, str]] = {}
	for case in os.listdir(results_dir):
		case_dir = os.path.join(results_dir, case)
		if not os.path.isdir(case_dir):
			continue
		for fname in os.listdir(case_dir):
			if not fname.endswith('.pth'):
				continue
			try:
				policy, parsed_case = parse_flat_filename(fname)
			except ValueError:
				continue
			index.setdefault(policy, {})[parsed_case] = os.path.join(case_dir, fname)
	return index


def handle_th_exp_ablation(results_dir: str, use_link: bool = False, dry_run: bool = False) -> List[str]:
	"""Create files for (d) th-exp-ablation.
	Outputs: results/th-exp-ablation/suika-th-exp-{lbd:.2f}.pth
	Sources: Prefer organized subfolders (results/lbdX.YY/suika-lbdX.YY.pth); if not present,
	fall back to flat files (results/suika-lbdX.YY.pth).
	"""
	logs: List[str] = []
	# Collect lbd sources from both subfolders and flat files
	sources: Dict[str, str] = {}
	# From flat files
	for entry in os.listdir(results_dir):
		m = re.match(r'^suika-(lbd\d+(?:\.\d+)?)\.pth$', entry)
		if m:
			case = m.group(1)
			sources[case] = os.path.join(results_dir, entry)
	# From subfolders (if any were created previously)
	for sub in os.listdir(results_dir):
		sub_dir = os.path.join(results_dir, sub)
		if not os.path.isdir(sub_dir) or not sub.startswith('lbd'):
			continue
		candidate = os.path.join(sub_dir, f"suika-{sub}.pth")
		if os.path.isfile(candidate):
			sources[sub] = candidate

	dst_dir = os.path.join(results_dir, 'th-exp-ablation')
	ensure_dir(dst_dir)

	for case, src_path in sources.items():
		m = re.match(r'^lbd(\d+(?:\.\d+)?)$', case)
		if not m:
			continue
		try:
			val = float(m.group(1))
		except ValueError:
			continue
		dst_name = f"suika-th-exp-{val:.2f}.pth"
		dst_path = os.path.join(dst_dir, dst_name)
		copy_or_link(src_path, dst_path, use_link, dry_run=dry_run)
		logs.append(f"(d) th-exp: {os.path.relpath(src_path, results_dir)} -> th-exp-ablation/{dst_name}")
	return logs


def find_first_available(index: Dict[str, Dict[str, str]], policy: str, preferred_cases: List[str]) -> Tuple[str, str]:
	"""Return (case, path) for the policy by preferred case order, else any available."""
	cases = index.get(policy, {})
	for c in preferred_cases:
		if c in cases:
			return c, cases[c]
	# fallback to any
	for c, p in cases.items():
		return c, p
	return '', ''


def handle_base_policy_ablation(results_dir: str, use_link: bool = False, dry_run: bool = False) -> List[str]:
	"""Create files for (e) base policy ablation.
	Expected:
	  - results/base-policy-ablation/suika-fcfs.pth
	  - results/base-policy-ablation/suika-sjf.pth
	  - results/base-policy-ablation/fcfs.pth (baseline FIFO)
	  - results/base-policy-ablation/sjf.pth  (baseline SJF)

	Also mirror into 'sjf-ablation' to satisfy notebook variants.
	Sources are discovered from organized case folders.
	"""
	logs: List[str] = []
	index = build_results_index(results_dir)

	base_dir = os.path.join(results_dir, 'base-policy-ablation')
	ensure_dir(base_dir)
	sjf_dir = os.path.join(results_dir, 'sjf-ablation')
	ensure_dir(sjf_dir)

	# suika-sjf comes from case 'sjf'
	suika_sjf_path = index.get('suika', {}).get('sjf')
	if suika_sjf_path:
		for d in (base_dir, sjf_dir):
			dst_path = os.path.join(d, 'suika-sjf.pth')
			copy_or_link(suika_sjf_path, dst_path, use_link, dry_run=dry_run)
			logs.append(f"(e) base: {os.path.relpath(suika_sjf_path, results_dir)} -> {os.path.relpath(dst_path, results_dir)}")

	# suika-fcfs is sourced from 'suika-clusterA'
	suika_fcfs_path = index.get('suika', {}).get('clusterA')
	if suika_fcfs_path:
		for d in (base_dir, sjf_dir):
			dst_path = os.path.join(d, 'suika-fcfs.pth')
			copy_or_link(suika_fcfs_path, dst_path, use_link, dry_run=dry_run)
			logs.append(f"(e) base: {os.path.relpath(suika_fcfs_path, results_dir)} -> {os.path.relpath(dst_path, results_dir)}")

	# fcfs baseline comes specifically from 'clusterA'
	fcfs_path = index.get('fcfs', {}).get('clusterA')
	if fcfs_path:
		for d in (base_dir, sjf_dir):
			dst_path = os.path.join(d, 'fcfs.pth')
			copy_or_link(fcfs_path, dst_path, use_link, dry_run=dry_run)
			logs.append(f"(e) base: {os.path.relpath(fcfs_path, results_dir)} -> {os.path.relpath(dst_path, results_dir)}")

	# sjf baseline comes specifically from 'clusterA'
	sjf_path = index.get('sjf', {}).get('clusterA')
	if sjf_path:
		for d in (base_dir, sjf_dir):
			dst_path = os.path.join(d, 'sjf.pth')
			copy_or_link(sjf_path, dst_path, use_link, dry_run=dry_run)
			logs.append(f"(e) base: {os.path.relpath(sjf_path, results_dir)} -> {os.path.relpath(dst_path, results_dir)}")

	return logs


def main():
	import argparse
	parser = argparse.ArgumentParser(description='Organize result files for visualization notebooks.')
	parser.add_argument('--results-dir', type=str, default=RESULTS_DIR_DEFAULT, help='Path to results directory')
	parser.add_argument('--link', action='store_true', help='Use symlinks instead of copying')
	parser.add_argument('--dry-run', action='store_true', help='Print actions without performing them')
	args = parser.parse_args()

	results_dir = os.path.abspath(args.results_dir)
	if not os.path.isdir(results_dir):
		raise SystemExit(f"Results directory not found: {results_dir}")

	# Discover flat files
	flat_files = [f for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f)) and f.endswith('.pth')]
	if not flat_files:
		print(f"No flat .pth files found directly under {results_dir}; nothing to organize.")
	else:
		print(f"Found {len(flat_files)} flat .pth file(s) in {results_dir}.")

	# Organize flat files
	org_logs = organize_flat_files(results_dir, use_link=args.link, dry_run=args.dry_run)
	for line in org_logs:
		print(line)


	# Handle special cases for sensitivity study
	th_logs = handle_th_exp_ablation(results_dir, use_link=args.link, dry_run=args.dry_run)
	for line in th_logs:
		print(line)
	base_logs = handle_base_policy_ablation(results_dir, use_link=args.link, dry_run=args.dry_run)
	for line in base_logs:
		print(line)

	print("Done.")


if __name__ == '__main__':
	main()
# goal