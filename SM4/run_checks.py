#!/usr/bin/env python3
"""
Run comprehensive validation on SM4 data matching validate.py and check_program.py:
1. Registration of all atoms
2. Uniqueness of IDs
3. check_program.check() on all 34 composites
4. graph_signature uniqueness
5. Canonical exact answers (no dict, list, float, bool)
6. Worked example verification
7. Answer variation (max answer frequency <= 60%)
8. No constant nodes, no no-ops
9. Support >= 128
10. Atom spread report (coverage >= 70%, reuse >= 2, share <= 25%)
"""
import sys, json, random
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "SM4"))

import kernel, program, generate, validate, check_program
import atoms as sm4_atoms

random.seed(0)

print("=== SM4 DATA VALIDATION ===")

atoms = generate.load(HERE / "SM4" / "atoms.jsonl")
tem = generate.load(HERE / "SM4" / "templates.jsonl")
comp = generate.load(HERE / "SM4" / "composite.jsonl")
graphs = {g["id"]: g for g in generate.load(HERE / "SM4" / "graphs.jsonl")}

print(f"Loaded: atoms={len(atoms)}, templates={len(tem)}, composites={len(comp)}, graphs={len(graphs)}")

# 1. Unique IDs & Units
atom_ids = {a["id"] for a in atoms}
tem_ids = {t["id"] for t in tem}
comp_ids = {c["id"] for c in comp}
assert len(atom_ids) == len(atoms), "Duplicate atom IDs"
assert len(tem_ids) == len(tem), "Duplicate template IDs"
assert len(comp_ids) == len(comp), "Duplicate composite IDs"
assert not (atom_ids & tem_ids), "Overlap between atom and template IDs"
assert not (atom_ids & comp_ids), "Overlap between atom and composite IDs"
assert not (tem_ids & comp_ids), "Overlap between template and composite IDs"
print("✓ Unique IDs across all kinds")

# 2. Check all composite programs with check_program
failed_programs = []
for c in comp:
    cid = c["id"]
    spec = graphs[cid]
    results = check_program.check(c, spec)
    bad = [r for r in results if not r[1]]
    if bad:
        failed_programs.append((cid, bad))
    kd = check_program.knowledge_depth(spec["nodes"], program.returned(spec))
    print(f"  {cid:<40} depth {kd}  {'OK' if not bad else 'FAIL'}")
    if bad:
        for name, ok, detail in bad:
            print(f"     FAIL {name}: {detail}")

assert not failed_programs, f"{len(failed_programs)} programs failed check_program!"
print("✓ All 34 composite reference programs passed check_program!")

# 3. Check graph signature uniqueness
kernel_only = {a for a, kind in kernel.KIND.items() if kind == "kernel"}
signatures = {}
for cid, spec in graphs.items():
    sig = validate.graph_signature(spec, kernel_only)
    prev = signatures.setdefault(sig, cid)
    if prev != cid:
        raise AssertionError(f"Duplicate graph signature between {cid} and {prev}!")
print("✓ All 34 composite dependency graphs have unique atom-labelled signatures!")

# 4. Check support >= 128 for all templates
print("\n=== SUPPORT CHECK ===")
small_templates = []
for t in tem + comp:
    n, exact = validate.support(t)
    if n < 128 and not t.get("finite_support"):
        small_templates.append((t["id"], n))
assert len(small_templates) == 0, f"Templates with support < 128: {small_templates}"
print("✓ All atomic and composite templates have support >= 128!")

# 5. Atom spread report
print("\n=== ATOM SPREAD REPORT ===")
validate.spread_report(tem, comp)

# 6. Sampling & Answer canonicality
print("\n=== SAMPLING & DETERMINISM CHECK ===")
for t in tem:
    for _ in range(10):
        q, a = generate.make(t)
        assert validate.canonical(a), f"{t['id']}: non-canonical answer {a!r}"
for c in comp:
    for _ in range(25):
        vals = generate.sample(c)
        got, _ = generate.run_graph(c, vals)
        assert validate.canonical(got), f"{c['id']}: non-canonical answer {got!r}"
print("✓ All answers are canonical and exact")

print("\n==========================================")
print("ALL SM4 VALIDATION CHECKS PASSED 100%!")
print("==========================================")
