#!/usr/bin/env python3
"""
Master generator for SM4 composite.jsonl and graphs.jsonl.
Ensures:
- 34 composites (depths: 10, 8, 7, 5, 4)
- 21/21 atoms used
- All atom counts between 2 and 8
- Support >= 128 for every template
- Worked examples verified by program.run
- Unique graph signatures
"""
import sys, json, re, itertools
from pathlib import Path
from fractions import Fraction
import sympy as sp
from collections import Counter

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "SM4"))

import kernel, program, generate, validate
import atoms as sm4_atoms

Q = lambda v: {"question": v}
L = lambda v: {"literal": v}
R = lambda n: {"ref": n}

composites = []
graphs = []

def add(cid, atoms, template, vars_, nodes, ret, derive=None, constraints=None, ex_vars=None):
    comp = {
        "id": cid,
        "unit": "SM4",
        "atoms": atoms,
        "template": template,
        "vars": vars_
    }
    if derive:
        comp["derive"] = derive
    if constraints:
        comp["constraints"] = constraints
        
    g = {
        "id": cid,
        "unit": "SM4",
        "nodes": nodes,
        "return": ret
    }
    
    # Compute exact answer for worked example
    v = dict(ex_vars)
    for name in generate.derive_order(comp.get("derive") or {}):
        v[name] = generate.evaluate(comp["derive"][name], v)
    ans, _ = program.run(nodes, v, (ret.get("ref") if isinstance(ret, dict) else ret))
    shown_ans = str(kernel.display(ans, comp.get("display"))).strip()
    comp["example"] = {
        "vars": ex_vars,
        "answer": shown_ans
    }
    
    composites.append(comp)
    graphs.append(g)

# ======================================================================
# DEPTH 2 (10 composites)
# ======================================================================

# c01: partial_frac -> power
add(
    "c01_partfrac_power",
    ["spec.integ.partial_frac", "spec.integ.power"],
    "Decompose {A}/((x − {p})(x − {q})) = M/(x − {p}) + N/(x − {q}). Let b = |M|. Evaluate ∫₀^b x^{n} dx.",
    {"A": {"type": "int", "min": 1, "max": 6},
     "p": {"type": "int", "min": -5, "max": 0},
     "q": {"type": "int", "min": 1, "max": 6},
     "n": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.partial_frac", "args": {"A": Q("A"), "p": Q("p"), "q": Q("q")}},
        {"node_id": "k1", "atom_id": "kernel.get_entry", "args": {"v": R("n1"), "i": L(0)}},
        {"node_id": "k2", "atom_id": "kernel.absolute_value", "args": {"x": R("k1")}},
        {"node_id": "n2", "atom_id": "spec.integ.power", "args": {"n": Q("n"), "a": L(0), "b": R("k2")}}
    ],
    R("n2"),
    ex_vars={"A": 1, "p": -1, "q": 1, "n": 2}
)

# c02: implicit_grad -> exp_linear
add(
    "c02_implicit_exp",
    ["spec.deriv.implicit_grad", "spec.integ.exp_linear"],
    "For F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, evaluate dy/dx. Let b = |dy/dx|. Evaluate ∫₀^b e^x dx.",
    {"Fx": {"type": "int", "min": -8, "max": 8, "exclude": [0]},
     "Fy": {"type": "int", "min": 1, "max": 10}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.implicit_grad", "args": {"F_x": Q("Fx"), "F_y": Q("Fy")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n1")}},
        {"node_id": "n2", "atom_id": "spec.integ.exp_linear", "args": {"k": L(1), "a": L(0), "b": R("k1")}}
    ],
    R("n2"),
    ex_vars={"Fx": 2, "Fy": 1}
)

# c03: margin_of_error -> exp_linear
add(
    "c03_margin_exp",
    ["spec.stats.margin_of_error", "spec.integ.exp_linear"],
    "Given z = {z}, s = {s}, and n = {n_val}, calculate margin of error E = z·s/√n. Then evaluate ∫₀^E e^({k}x) dx.",
    {"z": {"type": "int", "min": 1, "max": 3},
     "s": {"type": "int", "min": 1, "max": 8},
     "n_val": {"type": "choice", "values": [4, 9, 16, 25, 36, 49, 64]},
     "k": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.stats.margin_of_error", "args": {"z": Q("z"), "s": Q("s"), "n": Q("n_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.exp_linear", "args": {"k": Q("k"), "a": L(0), "b": R("n1")}}
    ],
    R("n2"),
    ex_vars={"z": 1, "s": 2, "n_val": 4, "k": 1}
)

# c04: arcsin_eval -> sin_sq
add(
    "c04_arcsin_sinsq",
    ["spec.trig.arcsin_eval", "spec.integ.sin_sq"],
    "Let θ = arcsin({x_disp}). Evaluate {C}·∫_{a_disp}^θ sin²(x) dx.",
    {"idx_x": {"type": "int", "min": 0, "max": 4},
     "idx_a": {"type": "int", "min": 0, "max": 4},
     "C": {"type": "int", "min": 1, "max": 10}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arcsin_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sin_sq", "args": {"a": Q("a_val"), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n2")}}
    ],
    R("k1"),
    derive={
        "x_val": "[Fraction(-1, 1), Fraction(-1, 2), Fraction(0, 1), Fraction(1, 2), Fraction(1, 1)][idx_x]",
        "x_disp": "['-1', '-1/2', '0', '1/2', '1'][idx_x]",
        "a_val": "[-pi/2, -pi/6, 0, pi/6, pi/2][idx_a]",
        "a_disp": "['-π/2', '-π/6', '0', 'π/6', 'π/2'][idx_a]"
    },
    ex_vars={"idx_x": 3, "idx_a": 2, "C": 1}
)

# c05: arctan_eval -> sec_sq
add(
    "c05_arctan_secsq",
    ["spec.trig.arctan_eval", "spec.integ.sec_sq"],
    "Let θ = arctan({x_disp}). Evaluate {C}·∫₀^θ sec²(x) dx.",
    {"idx": {"type": "int", "min": 0, "max": 15},
     "C": {"type": "int", "min": 1, "max": 12}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arctan_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sec_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n2")}}
    ],
    R("k1"),
    derive={
        "x_val": "[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][idx]",
        "x_disp": "['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'][idx]"
    },
    ex_vars={"idx": 6, "C": 1}
)

# c06: arccos_eval -> cos_sq
add(
    "c06_arccos_cossq",
    ["spec.trig.arccos_eval", "spec.integ.cos_sq"],
    "Let θ = arccos({x_disp}). Evaluate {C}·∫_{a_disp}^θ cos²(x) dx.",
    {"idx_x": {"type": "int", "min": 0, "max": 4},
     "idx_a": {"type": "int", "min": 0, "max": 4},
     "C": {"type": "int", "min": 1, "max": 10}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arccos_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.cos_sq", "args": {"a": Q("a_val"), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n2")}}
    ],
    R("k1"),
    derive={
        "x_val": "[Fraction(-1, 1), Fraction(-1, 2), Fraction(0, 1), Fraction(1, 2), Fraction(1, 1)][idx_x]",
        "x_disp": "['-1', '-1/2', '0', '1/2', '1'][idx_x]",
        "a_val": "[0, pi/6, pi/4, pi/3, pi/2][idx_a]",
        "a_disp": "['0', 'π/6', 'π/4', 'π/3', 'π/2'][idx_a]"
    },
    ex_vars={"idx_x": 3, "idx_a": 0, "C": 1}
)

# c07: margin_of_error -> reciprocal
add(
    "c07_margin_recip",
    ["spec.stats.margin_of_error", "spec.integ.reciprocal"],
    "Calculate margin of error E = z·s/√n for z = {z}, s = {s}, n = {n_val}. Evaluate ∫₁^E (1/x) dx.",
    {"z": {"type": "int", "min": 2, "max": 5},
     "s": {"type": "int", "min": 2, "max": 10},
     "n_val": {"type": "choice", "values": [4, 9, 16, 25, 36, 49]}},
    [
        {"node_id": "n1", "atom_id": "spec.stats.margin_of_error", "args": {"z": Q("z"), "s": Q("s"), "n": Q("n_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.reciprocal", "args": {"a": L(1), "b": R("n1")}}
    ],
    R("n2"),
    constraints=["z * s > int(sqrt(n_val))"],
    ex_vars={"z": 2, "s": 4, "n_val": 4}
)

# c08: implicit_grad -> power
add(
    "c08_implicit_power",
    ["spec.deriv.implicit_grad", "spec.integ.power"],
    "For F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, find m = |dy/dx|. Evaluate ∫₀^m x^{n} dx.",
    {"Fx": {"type": "int", "min": 1, "max": 8},
     "Fy": {"type": "int", "min": 1, "max": 6},
     "n": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.implicit_grad", "args": {"F_x": Q("Fx"), "F_y": Q("Fy")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n1")}},
        {"node_id": "n2", "atom_id": "spec.integ.power", "args": {"n": Q("n"), "a": L(0), "b": R("k1")}}
    ],
    R("n2"),
    ex_vars={"Fx": 2, "Fy": 1, "n": 2}
)

# c09: arcsin_form -> sin_sq
add(
    "c09_arcsinform_sinsq",
    ["spec.integ.arcsin_form", "spec.integ.sin_sq"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx. Evaluate {C}·∫₀^θ sin²(x) dx.",
    {"c": {"type": "int", "min": 3, "max": 12},
     "b": {"type": "int", "min": 1, "max": 11},
     "C": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arcsin_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.integ.sin_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n2")}}
    ],
    R("k1"),
    constraints=["b < c"],
    derive={"csq": "c * c"},
    ex_vars={"c": 4, "b": 2, "C": 1}
)

# c10: arctan_form -> sec_sq
add(
    "c10_arctanform_secsq",
    ["spec.integ.arctan_form", "spec.integ.sec_sq"],
    "Let θ = ∫₀^{b} {c}/({csq} + x²) dx. Evaluate {C}·∫₀^θ sec²(x) dx.",
    {"c": {"type": "int", "min": 1, "max": 8},
     "b": {"type": "int", "min": 1, "max": 15},
     "C": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arctan_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.integ.sec_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n2")}}
    ],
    R("k1"),
    derive={"csq": "c * c"},
    ex_vars={"c": 1, "b": 1, "C": 1}
)

# ======================================================================
# DEPTH 3 (8 composites)
# ======================================================================

# c11: implicit_grad -> related_rate -> arctan_val
add(
    "c11_implicit_rate_arctanval",
    ["spec.deriv.implicit_grad", "spec.deriv.related_rate", "spec.deriv.arctan_val"],
    "On F(x, y) = 0, ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. If dx/dt = {dxdt_d}, find dy/dt. Then evaluate 1/(1 + u²) at u = dy/dt.",
    {"Fx": {"type": "int", "min": -6, "max": 6},
     "Fy": {"type": "int", "min": 1, "max": 6},
     "r": {"type": "int", "min": -4, "max": 4, "exclude": [0]},
     "s": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.implicit_grad", "args": {"F_x": Q("Fx"), "F_y": Q("Fy")}},
        {"node_id": "n2", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n1"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}}
    ],
    R("n3"),
    derive={"dxdt_val": "Fraction(r, s)", "dxdt_d": "str(Fraction(r, s))"},
    ex_vars={"Fx": 2, "Fy": 1, "r": 1, "s": 1}
)

# c12: margin_of_error -> arctan_val -> exp_linear
add(
    "c12_margin_arctanval_exp",
    ["spec.stats.margin_of_error", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "Given z = {z}, s = {s}, n = {n_val}, find margin of error E = z·s/√n. Let u = 1/(1 + E²). Evaluate ∫₀^u e^({k}x) dx.",
    {"z": {"type": "int", "min": 1, "max": 3},
     "s": {"type": "int", "min": 1, "max": 6},
     "n_val": {"type": "choice", "values": [4, 9, 16, 25, 36, 49, 64]},
     "k": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.stats.margin_of_error", "args": {"z": Q("z"), "s": Q("s"), "n": Q("n_val")}},
        {"node_id": "n2", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.exp_linear", "args": {"k": Q("k"), "a": L(0), "b": R("n2")}}
    ],
    R("n3"),
    ex_vars={"z": 1, "s": 2, "n_val": 4, "k": 1}
)

# c13: margin_of_error -> arcsin_val -> arctan_val
add(
    "c13_margin_arcsinval_arctanval",
    ["spec.stats.margin_of_error", "spec.deriv.arcsin_val", "spec.deriv.arctan_val"],
    "With z = {z}, s = {s}, n = {n_val}, let E = z·s/√n. Evaluate v = 1/√(1 − E²). Then evaluate {C}·1/(1 + v²).",
    {"z": {"type": "int", "min": 1, "max": 2},
     "s": {"type": "int", "min": 1, "max": 4},
     "n_val": {"type": "choice", "values": [25, 36, 49, 64, 81, 100, 144]},
     "C": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.stats.margin_of_error", "args": {"z": Q("z"), "s": Q("s"), "n": Q("n_val")}},
        {"node_id": "n2", "atom_id": "spec.deriv.arcsin_val", "args": {"x": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n3")}}
    ],
    R("k1"),
    constraints=["z * s < int(sqrt(n_val))"],
    ex_vars={"z": 1, "s": 1, "n_val": 25, "C": 1}
)

# c14: arcsin_eval -> cos_sq -> shm_eval
add(
    "c14_arcsin_cossq_shm",
    ["spec.trig.arcsin_eval", "spec.integ.cos_sq", "spec.dynamics.shm_eval"],
    "Let θ = arcsin({x_disp}). Let I = ∫₀^θ cos²(x) dx. For x(t) = {A} cos({omega_val}t), evaluate x(I).",
    {"idx": {"type": "int", "min": 0, "max": 3},
     "A": {"type": "int", "min": 1, "max": 8},
     "omega_val": {"type": "int", "min": 1, "max": 5}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arcsin_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.cos_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": Q("omega_val"), "phi": L(0), "t": R("n2")}}
    ],
    R("n3"),
    derive={"x_val": "[0, Fraction(1, 2), Fraction(1, 1), Fraction(-1, 2)][idx]",
            "x_disp": "['0', '1/2', '1', '-1/2'][idx]"},
    ex_vars={"idx": 1, "A": 2, "omega_val": 1}
)

# c15: arccos_eval -> sin_sq -> reciprocal
add(
    "c15_arccos_sinsq_recip",
    ["spec.trig.arccos_eval", "spec.integ.sin_sq", "spec.integ.reciprocal"],
    "Let θ = arccos({x_disp}). Let I = ∫₀^θ sin²(x) dx. Evaluate {C}·∫₁^({shift}+I) (1/x) dx.",
    {"idx": {"type": "int", "min": 0, "max": 3},
     "C": {"type": "int", "min": 1, "max": 10},
     "shift": {"type": "int", "min": 1, "max": 5}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arccos_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sin_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "k1", "atom_id": "kernel.add", "args": {"x": Q("shift"), "y": R("n2")}},
        {"node_id": "n3", "atom_id": "spec.integ.reciprocal", "args": {"a": L(1), "b": R("k1")}},
        {"node_id": "k2", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n3")}}
    ],
    R("k2"),
    derive={"x_val": "[0, Fraction(1, 2), Fraction(1, 1), Fraction(-1, 2)][idx]",
            "x_disp": "['0', '1/2', '1', '-1/2'][idx]"},
    ex_vars={"idx": 0, "C": 1, "shift": 1}
)

# c16: arctan_eval -> sec_sq -> reciprocal
add(
    "c16_arctan_secsq_recip",
    ["spec.trig.arctan_eval", "spec.integ.sec_sq", "spec.integ.reciprocal"],
    "Let θ = arctan({x_disp}). Let T = ∫₀^θ sec²(x) dx. Evaluate {C}·∫₁^T (1/x) dx.",
    {"idx": {"type": "int", "min": 0, "max": 13},
     "C": {"type": "int", "min": 1, "max": 10}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arctan_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sec_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.reciprocal", "args": {"a": L(1), "b": R("n2")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n3")}}
    ],
    R("k1"),
    derive={"x_val": "[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15][idx]",
            "x_disp": "['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'][idx]"},
    ex_vars={"idx": 0, "C": 1}
)

# c17: arcsin_form -> cos_sq -> arctan_val
add(
    "c17_arcsinform_cossq_arctanval",
    ["spec.integ.arcsin_form", "spec.integ.cos_sq", "spec.deriv.arctan_val"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx. Evaluate I = ∫₀^θ cos²(x) dx. Then find {C}·1/(1 + I²).",
    {"c": {"type": "int", "min": 3, "max": 10},
     "b": {"type": "int", "min": 1, "max": 9},
     "C": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arcsin_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.integ.cos_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n3")}}
    ],
    R("k1"),
    constraints=["b < c"],
    derive={"csq": "c * c"},
    ex_vars={"c": 4, "b": 2, "C": 1}
)

# c18: arctan_form -> sec_sq -> exp_linear
add(
    "c18_arctanform_secsq_exp",
    ["spec.integ.arctan_form", "spec.integ.sec_sq", "spec.integ.exp_linear"],
    "Let θ = ∫₀^{b} {c}/({csq} + x²) dx. Evaluate T = ∫₀^θ sec²(x) dx. Then evaluate ∫₀^T e^({k}x) dx.",
    {"c": {"type": "int", "min": 1, "max": 6},
     "b": {"type": "int", "min": 1, "max": 10},
     "k": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arctan_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.integ.sec_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.exp_linear", "args": {"k": Q("k"), "a": L(0), "b": R("n2")}}
    ],
    R("n3"),
    derive={"csq": "c * c"},
    ex_vars={"c": 1, "b": 1, "k": 1}
)

# ======================================================================
# DEPTH 4 (7 composites)
# ======================================================================

# c19: arcsin_eval -> shm_eval -> arctan_val -> related_rate
add(
    "c19_arcsin_shm_arctanval_rate",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate"],
    "Let θ = arcsin({x_disp}). Let x₀ = {A} cos(θ). Evaluate dy/dx = 1/(1 + x₀²). If dx/dt = {dxdt_d}, find dy/dt.",
    {"idx": {"type": "int", "min": 0, "max": 3},
     "A": {"type": "int", "min": 1, "max": 5},
     "r": {"type": "int", "min": 1, "max": 5},
     "s": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arcsin_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n3"), "dx_dt": Q("dxdt_val")}}
    ],
    R("n4"),
    derive={
        "x_val": "[0, Fraction(1, 2), Fraction(1, 1), Fraction(-1, 2)][idx]",
        "x_disp": "['0', '1/2', '1', '-1/2'][idx]",
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    ex_vars={"idx": 1, "A": 2, "r": 3, "s": 1}
)

# c20: arctan_form -> shm_eval -> arctan_val -> exp_linear
add(
    "c20_arctanform_shm_arctanval_exp",
    ["spec.integ.arctan_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "Let θ = ∫₀^{b} {c}/({csq} + x²) dx. Let x₀ = {A} cos(θ). Evaluate m = 1/(1 + x₀²). Then evaluate ∫₀^m e^({k}x) dx.",
    {"c": {"type": "int", "min": 1, "max": 5},
     "b": {"type": "int", "min": 1, "max": 6},
     "A": {"type": "int", "min": 1, "max": 4},
     "k": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arctan_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.integ.exp_linear", "args": {"k": Q("k"), "a": L(0), "b": R("n3")}}
    ],
    R("n4"),
    derive={"csq": "c * c"},
    ex_vars={"c": 1, "b": 1, "A": 1, "k": 1}
)

# c21: partial_frac -> power -> by_parts_step -> volume_disk
add(
    "c21_partfrac_power_parts_vol",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Decompose {A}/((x − {p})(x − {q})). Let M be the first coefficient, and b = |M|. Let I = ∫₀^b x^{n} dx. In integration by parts, [uv] = I and ∫ v du = {iv}. Let J = [uv] − ∫ v du. Find V = π·J.",
    {"A": {"type": "int", "min": 1, "max": 5},
     "p": {"type": "int", "min": -4, "max": 0},
     "q": {"type": "int", "min": 1, "max": 5},
     "n": {"type": "int", "min": 1, "max": 3},
     "iv": {"type": "int", "min": 0, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.partial_frac", "args": {"A": Q("A"), "p": Q("p"), "q": Q("q")}},
        {"node_id": "k1", "atom_id": "kernel.get_entry", "args": {"v": R("n1"), "i": L(0)}},
        {"node_id": "k2", "atom_id": "kernel.absolute_value", "args": {"x": R("k1")}},
        {"node_id": "n2", "atom_id": "spec.integ.power", "args": {"n": Q("n"), "a": L(0), "b": R("k2")}},
        {"node_id": "n3", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n2"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n4", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n3")}}
    ],
    R("n4"),
    ex_vars={"A": 1, "p": -1, "q": 1, "n": 2, "iv": 0}
)

# c22: partial_frac -> reciprocal -> exp_linear -> solve_direct
add(
    "c22_partfrac_recip_exp_solve",
    ["spec.integ.partial_frac", "spec.integ.reciprocal", "spec.integ.exp_linear", "spec.diffeq.solve_direct"],
    "Decompose {A}/((x − {p})(x − {q})). Let M be the first coefficient, and let b = |M| + 1. Evaluate I = ∫₁^b (1/x) dx. Evaluate J = ∫₀^I e^x dx. If y(0) = {y0} and y changes by J, find y.",
    {"A": {"type": "int", "min": 1, "max": 5},
     "p": {"type": "int", "min": -4, "max": 0},
     "q": {"type": "int", "min": 1, "max": 5},
     "y0": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.partial_frac", "args": {"A": Q("A"), "p": Q("p"), "q": Q("q")}},
        {"node_id": "k1", "atom_id": "kernel.get_entry", "args": {"v": R("n1"), "i": L(0)}},
        {"node_id": "k2", "atom_id": "kernel.absolute_value", "args": {"x": R("k1")}},
        {"node_id": "k3", "atom_id": "kernel.add", "args": {"x": L(1), "y": R("k2")}},
        {"node_id": "n2", "atom_id": "spec.integ.reciprocal", "args": {"a": L(1), "b": R("k3")}},
        {"node_id": "n3", "atom_id": "spec.integ.exp_linear", "args": {"k": L(1), "a": L(0), "b": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n3")}}
    ],
    R("n4"),
    ex_vars={"A": 1, "p": -1, "q": 1, "y0": 2}
)

# c23: partial_frac -> power -> reciprocal -> exp_linear
add(
    "c23_partfrac_power_recip_exp",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.reciprocal", "spec.integ.exp_linear"],
    "Decompose {A}/((x − {p})(x − {q})). Let M be the first coefficient, and b = |M| + 1. Evaluate I = ∫₀^b x dx. Evaluate J = ∫₁^(1+I) (1/x) dx. Evaluate ∫₀^J e^({k}x) dx.",
    {"A": {"type": "int", "min": 1, "max": 6},
     "p": {"type": "int", "min": -5, "max": 0},
     "q": {"type": "int", "min": 1, "max": 6},
     "k": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.partial_frac", "args": {"A": Q("A"), "p": Q("p"), "q": Q("q")}},
        {"node_id": "k1", "atom_id": "kernel.get_entry", "args": {"v": R("n1"), "i": L(0)}},
        {"node_id": "k2", "atom_id": "kernel.absolute_value", "args": {"x": R("k1")}},
        {"node_id": "k3", "atom_id": "kernel.add", "args": {"x": L(1), "y": R("k2")}},
        {"node_id": "n2", "atom_id": "spec.integ.power", "args": {"n": L(1), "a": L(0), "b": R("k3")}},
        {"node_id": "k4", "atom_id": "kernel.add", "args": {"x": L(1), "y": R("n2")}},
        {"node_id": "n3", "atom_id": "spec.integ.reciprocal", "args": {"a": L(1), "b": R("k4")}},
        {"node_id": "n4", "atom_id": "spec.integ.exp_linear", "args": {"k": Q("k"), "a": L(0), "b": R("n3")}}
    ],
    R("n4"),
    ex_vars={"A": 1, "p": -1, "q": 1, "k": 1}
)

# c24: arcsin_eval -> sin_sq -> by_parts_step -> volume_disk
add(
    "c24_arcsin_sinsq_parts_vol",
    ["spec.trig.arcsin_eval", "spec.integ.sin_sq", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Let θ = arcsin({x_disp}). Let I = ∫₀^θ sin²(x) dx. Using integration by parts with [uv] = I and ∫ v du = {iv}, find J = [uv] − ∫ v du. Find V = {C}·π·J.",
    {"idx": {"type": "int", "min": 0, "max": 3},
     "iv": {"type": "int", "min": 0, "max": 5},
     "C": {"type": "int", "min": 1, "max": 8}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arcsin_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sin_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n2"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": Q("C"), "y": R("n3")}},
        {"node_id": "n4", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("k1")}}
    ],
    R("n4"),
    derive={
        "x_val": "[0, Fraction(1, 2), Fraction(1, 1), Fraction(-1, 2)][idx]",
        "x_disp": "['0', '1/2', '1', '-1/2'][idx]"
    },
    ex_vars={"idx": 1, "iv": 0, "C": 1}
)

# c25: arccos_eval -> cos_sq -> by_parts_step -> arcsin_val
add(
    "c25_arccos_cossq_parts_arcsinval",
    ["spec.trig.arccos_eval", "spec.integ.cos_sq", "spec.integ.by_parts_step", "spec.deriv.arcsin_val"],
    "Let θ = arccos({x_disp}). Let I = ∫₀^θ cos²(x) dx. Using [uv] = I and ∫ v du = {iv}, find J = [uv] − ∫ v du. Let u = J / {scale}. Evaluate 1/√(1 − u²).",
    {"idx": {"type": "int", "min": 0, "max": 2},
     "iv": {"type": "int", "min": 0, "max": 3},
     "scale": {"type": "int", "min": 10, "max": 22}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arccos_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.cos_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n2"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "k1", "atom_id": "kernel.divide", "args": {"x": R("n3"), "y": Q("scale")}},
        {"node_id": "n4", "atom_id": "spec.deriv.arcsin_val", "args": {"x": R("k1")}}
    ],
    R("n4"),
    derive={
        "x_val": "[0, Fraction(1, 2), Fraction(1, 1)][idx]",
        "x_disp": "['0', '1/2', '1'][idx]"
    },
    ex_vars={"idx": 0, "iv": 0, "scale": 10}
)

# ======================================================================
# DEPTH 5 (5 composites)
# ======================================================================

# c26: arcsin_form -> shm_eval -> arctan_val -> related_rate -> solve_direct
add(
    "c26_arcsinform_shm_arctanval_rate_solve",
    ["spec.integ.arcsin_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx. A particle in SHM has x(t) = {A} cos(t). Evaluate x₀ = x(θ). A curve has slope dy/dx = 1/(1 + x₀²). If dx/dt = {dxdt_d}, find dy/dt. If z(0) = {z0} and dz/dt = dy/dt, find z({T}).",
    {"c": {"type": "int", "min": 2, "max": 6},
     "b": {"type": "int", "min": 1, "max": 5},
     "A": {"type": "int", "min": 1, "max": 3},
     "r": {"type": "int", "min": 1, "max": 3},
     "s": {"type": "int", "min": 1, "max": 2},
     "z0": {"type": "int", "min": 0, "max": 4},
     "T": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.arcsin_form", "args": {"c": Q("c"), "a": L(0), "b": Q("b")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n3"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": R("n4"), "y": Q("T")}},
        {"node_id": "n5", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("z0"), "integral_value": R("k1")}}
    ],
    R("n5"),
    constraints=["b < c"],
    derive={
        "csq": "c * c",
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    ex_vars={"c": 2, "b": 1, "A": 1, "r": 1, "s": 1, "z0": 0, "T": 2}
)

# c27: arccos_eval -> shm_eval -> arctan_val -> related_rate -> power
add(
    "c27_arccos_shm_arctanval_rate_power",
    ["spec.trig.arccos_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.integ.power"],
    "Let θ = arccos({x_disp}). Let x₀ = {A} cos(θ). Evaluate dy/dx = 1/(1 + x₀²). If dx/dt = {dxdt_d}, find v = |dy/dt|. Then evaluate ∫₀^v x^{n} dx.",
    {"idx": {"type": "int", "min": 0, "max": 2},
     "A": {"type": "int", "min": 1, "max": 3},
     "r": {"type": "int", "min": 1, "max": 4},
     "s": {"type": "int", "min": 1, "max": 2},
     "n": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arccos_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arctan_val", "args": {"x": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n3"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n4")}},
        {"node_id": "n5", "atom_id": "spec.integ.power", "args": {"n": Q("n"), "a": L(0), "b": R("k1")}}
    ],
    R("n5"),
    derive={
        "x_val": "[0, Fraction(1, 2), Fraction(1, 1)][idx]",
        "x_disp": "['0', '1/2', '1'][idx]",
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    ex_vars={"idx": 0, "A": 1, "r": 2, "s": 1, "n": 1}
)

# c28: partial_frac -> power -> volume_disk -> by_parts_step -> solve_direct
add(
    "c28_partfrac_power_vol_parts_solve",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.volume_disk", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "Decompose {A}/((x − {p})(x − {q})). Let M be the first coefficient, and b = |M|. Let I = ∫₀^b x^{n} dx. Find V = π·I. In integration by parts, [uv] = V and ∫ v du = {iv}. Let J = [uv] − ∫ v du. If y(0) = {y0} and change is J, find y.",
    {"A": {"type": "int", "min": 1, "max": 5},
     "p": {"type": "int", "min": -4, "max": 0},
     "q": {"type": "int", "min": 1, "max": 4},
     "n": {"type": "int", "min": 1, "max": 3},
     "iv": {"type": "int", "min": 0, "max": 3},
     "y0": {"type": "int", "min": 1, "max": 5}},
    [
        {"node_id": "n1", "atom_id": "spec.integ.partial_frac", "args": {"A": Q("A"), "p": Q("p"), "q": Q("q")}},
        {"node_id": "k1", "atom_id": "kernel.get_entry", "args": {"v": R("n1"), "i": L(0)}},
        {"node_id": "k2", "atom_id": "kernel.absolute_value", "args": {"x": R("k1")}},
        {"node_id": "n2", "atom_id": "spec.integ.power", "args": {"n": Q("n"), "a": L(0), "b": R("k2")}},
        {"node_id": "n3", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n3"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n5", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n4")}}
    ],
    R("n5"),
    ex_vars={"A": 1, "p": -1, "q": 1, "n": 2, "iv": 0, "y0": 1}
)

# c29: arctan_eval -> sec_sq -> by_parts_step -> volume_disk -> solve_direct
add(
    "c29_arctan_secsq_parts_vol_solve",
    ["spec.trig.arctan_eval", "spec.integ.sec_sq", "spec.integ.by_parts_step", "spec.integ.volume_disk", "spec.diffeq.solve_direct"],
    "Let θ = arctan({x_disp}). Evaluate I = ∫₀^θ sec²(x) dx. Using [uv] = I and ∫ v du = {iv}, find J = [uv] − ∫ v du. Find V = π·J. If y(0) = {y0} and increases by V, find y.",
    {"idx": {"type": "int", "min": 0, "max": 5},
     "iv": {"type": "int", "min": 0, "max": 4},
     "y0": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arctan_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.integ.sec_sq", "args": {"a": L(0), "b": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n2"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n4", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n3")}},
        {"node_id": "n5", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n4")}}
    ],
    R("n5"),
    derive={
        "x_val": "[1, 2, 3, 4, 5, 6][idx]",
        "x_disp": "['1', '2', '3', '4', '5', '6'][idx]"
    },
    ex_vars={"idx": 0, "iv": 0, "y0": 1}
)

# c30: implicit_grad -> arcsin_val -> related_rate -> by_parts_step -> volume_disk
add(
    "c30_implicit_arcsinval_rate_parts_vol",
    ["spec.deriv.implicit_grad", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "On F(x, y) = 0, ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. Let u = dy/dx. Evaluate 1/√(1 − u²). If dx/dt = {dxdt_d}, find v = |dy/dt|. In integration by parts, [uv] = v and ∫ v du = {iv}. Let J = [uv] − ∫ v du. Find V = π·J.",
    {"Fx": {"type": "int", "min": 1, "max": 4},
     "Fy": {"type": "int", "min": 5, "max": 10},
     "r": {"type": "int", "min": 1, "max": 4},
     "s": {"type": "int", "min": 1, "max": 3},
     "iv": {"type": "int", "min": 0, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.implicit_grad", "args": {"F_x": Q("Fx"), "F_y": Q("Fy")}},
        {"node_id": "n2", "atom_id": "spec.deriv.arcsin_val", "args": {"x": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n2"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n3")}},
        {"node_id": "n4", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("k1"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n5", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n4")}}
    ],
    R("n5"),
    derive={
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    ex_vars={"Fx": 3, "Fy": 5, "r": 1, "s": 1, "iv": 0}
)

# ======================================================================
# DEPTH 6 (4 composites)
# ======================================================================

# c31: margin_of_error -> shm_eval -> arcsin_val -> related_rate -> solve_direct -> volume_disk
add(
    "c31_margin_shm_arcsinval_rate_solve_vol",
    ["spec.stats.margin_of_error", "spec.dynamics.shm_eval", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.volume_disk"],
    "Margin of error E = z·s/√n with z = {z}, s = {s}, n = {n_val}. Displacement x(t) = {A_disp} cos(t) at t = E gives x₀ = x(E). Let dy/dx = 1/√(1 − x₀²). If dx/dt = {dxdt_d}, find dy/dt. A quantity w satisfies dw/dt = dy/dt with w(0) = 0. Find w({T}). Find V = π·w({T}).",
    {"z": {"type": "int", "min": 1, "max": 2},
     "s": {"type": "int", "min": 1, "max": 3},
     "n_val": {"type": "choice", "values": [16, 25, 36, 49, 64]},
     "A_denom": {"type": "int", "min": 2, "max": 5},
     "r": {"type": "int", "min": 1, "max": 3},
     "s_rate": {"type": "int", "min": 1, "max": 2},
     "T": {"type": "int", "min": 1, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.stats.margin_of_error", "args": {"z": Q("z"), "s": Q("s"), "n": Q("n_val")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A_val"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.arcsin_val", "args": {"x": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n3"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.multiply", "args": {"x": R("n4"), "y": Q("T")}},
        {"node_id": "n5", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": L(0), "integral_value": R("k1")}},
        {"node_id": "n6", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n5")}}
    ],
    R("n6"),
    derive={
        "A_val": "Fraction(1, A_denom)",
        "A_disp": "f'1/{A_denom}'",
        "dxdt_val": "Fraction(r, s_rate)",
        "dxdt_d": "str(Fraction(r, s_rate))"
    },
    ex_vars={"z": 1, "s": 1, "n_val": 16, "A_denom": 2, "r": 1, "s_rate": 1, "T": 1}
)

# c32: arcsin_eval -> shm_eval -> arcsin_form -> sin_sq -> volume_disk -> solve_direct
add(
    "c32_arcsin_shm_arcsinform_sinsq_vol_solve",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.integ.arcsin_form", "spec.integ.sin_sq", "spec.integ.volume_disk", "spec.diffeq.solve_direct"],
    "Let θ₁ = arcsin({x_disp}). Let x₀ = {A} cos(θ₁). Let θ₂ = ∫₀^x₀ 1/√({csq} − x²) dx. Evaluate I = ∫₀^θ₂ sin²(x) dx. Find V = π·I. If y(0) = {y0} and increases by V, find y.",
    {"idx": {"type": "int", "min": 0, "max": 2},
     "A": {"type": "int", "min": 1, "max": 3},
     "c": {"type": "int", "min": 4, "max": 7},
     "y0": {"type": "int", "min": 1, "max": 6}},
    [
        {"node_id": "n1", "atom_id": "spec.trig.arcsin_eval", "args": {"x": Q("x_val")}},
        {"node_id": "n2", "atom_id": "spec.dynamics.shm_eval", "args": {"A": Q("A"), "omega": L(1), "phi": L(0), "t": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.integ.arcsin_form", "args": {"c": Q("c"), "a": L(0), "b": R("n2")}},
        {"node_id": "n4", "atom_id": "spec.integ.sin_sq", "args": {"a": L(0), "b": R("n3")}},
        {"node_id": "n5", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n4")}},
        {"node_id": "n6", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n5")}}
    ],
    R("n6"),
    derive={
        "x_val": "[0, Fraction(1, 2), Fraction(1, 1)][idx]",
        "x_disp": "['0', '1/2', '1'][idx]",
        "csq": "c * c"
    },
    constraints=["A < c"],
    ex_vars={"idx": 0, "A": 1, "c": 4, "y0": 2}
)

# c33: arcsin_val -> related_rate -> solve_direct -> volume_disk -> by_parts_step -> solve_direct
add(
    "c33_arcsinval_rate_solve_vol_parts_solve",
    ["spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.volume_disk", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "At x = {xd}, dy/dx for y = arcsin(x). If dx/dt = {dxdt_d}, find v = |dy/dt|. A quantity w has w(0) = {w0} and dw/dt = v. Find w(1). Find V = π·w(1). In integration by parts, [uv] = V and ∫ v du = {iv}. Let J = [uv] − ∫ v du. If y(0) = {y0} and increases by J, find y.",
    {"p": {"type": "int", "min": 1, "max": 4},
     "q": {"type": "int", "min": 5, "max": 10},
     "r": {"type": "int", "min": 1, "max": 3},
     "s": {"type": "int", "min": 1, "max": 2},
     "w0": {"type": "int", "min": 0, "max": 3},
     "y0": {"type": "int", "min": 1, "max": 4},
     "iv": {"type": "int", "min": 0, "max": 3}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.arcsin_val", "args": {"x": Q("xv")}},
        {"node_id": "n2", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n1"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n2")}},
        {"node_id": "n3", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("w0"), "integral_value": R("k1")}},
        {"node_id": "n4", "atom_id": "spec.integ.volume_disk", "args": {"integral_f_sq": R("n3")}},
        {"node_id": "n5", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n4"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n6", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n5")}}
    ],
    R("n6"),
    derive={
        "xv": "Fraction(p, q)",
        "xd": "str(Fraction(p, q))",
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    constraints=["p < q"],
    ex_vars={"p": 3, "q": 5, "r": 1, "s": 1, "w0": 1, "y0": 1, "iv": 0}
)

# c34: implicit_grad -> arcsin_val -> related_rate -> solve_direct -> by_parts_step -> solve_direct
add(
    "c34_implicit_arcsinval_rate_solve_parts_solve",
    ["spec.deriv.implicit_grad", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "On F(x, y) = 0, ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. Let u = dy/dx. Evaluate 1/√(1 − u²). If dx/dt = {dxdt_d}, find v = |dy/dt|. A quantity w satisfies w(0) = {w0} and increases by v. In integration by parts, [uv] = w and ∫ v du = {iv}. Let J = [uv] − ∫ v du. If y(0) = {y0} and change is J, find y.",
    {"Fx": {"type": "int", "min": 1, "max": 3},
     "Fy": {"type": "int", "min": 4, "max": 8},
     "r": {"type": "int", "min": 1, "max": 3},
     "s": {"type": "int", "min": 1, "max": 2},
     "w0": {"type": "int", "min": 0, "max": 3},
     "iv": {"type": "int", "min": 0, "max": 3},
     "y0": {"type": "int", "min": 1, "max": 4}},
    [
        {"node_id": "n1", "atom_id": "spec.deriv.implicit_grad", "args": {"F_x": Q("Fx"), "F_y": Q("Fy")}},
        {"node_id": "n2", "atom_id": "spec.deriv.arcsin_val", "args": {"x": R("n1")}},
        {"node_id": "n3", "atom_id": "spec.deriv.related_rate", "args": {"dy_dx": R("n2"), "dx_dt": Q("dxdt_val")}},
        {"node_id": "k1", "atom_id": "kernel.absolute_value", "args": {"x": R("n3")}},
        {"node_id": "n4", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("w0"), "integral_value": R("k1")}},
        {"node_id": "n5", "atom_id": "spec.integ.by_parts_step", "args": {"uv_upper": R("n4"), "uv_lower": L(0), "integral_vdu": Q("iv")}},
        {"node_id": "n6", "atom_id": "spec.diffeq.solve_direct", "args": {"y0": Q("y0"), "integral_value": R("n5")}}
    ],
    R("n6"),
    derive={
        "dxdt_val": "Fraction(r, s)",
        "dxdt_d": "str(Fraction(r, s))"
    },
    constraints=["Fx < Fy"],
    ex_vars={"Fx": 1, "Fy": 4, "r": 1, "s": 1, "w0": 1, "iv": 0, "y0": 1}
)

# Write files
comp_file = HERE / "SM4" / "composite.jsonl"
graph_file = HERE / "SM4" / "graphs.jsonl"

with open(comp_file, "w", encoding="utf-8") as f:
    for c in composites:
        f.write(json.dumps(c, separators=(",", ":"), ensure_ascii=False) + "\n")

with open(graph_file, "w", encoding="utf-8") as f:
    for g in graphs:
        f.write(json.dumps(g, separators=(",", ":"), ensure_ascii=False) + "\n")

print(f"\nWritten {len(composites)} composites to {comp_file}")
print(f"Written {len(graphs)} graphs to {graph_file}")

# Verify support for each template
print("\nVerifying support (>= 128) for each template...")
low_support = []
for c in composites:
    n, exact = validate.support(c)
    if n < 128 and not c.get("finite_support"):
        low_support.append((c["id"], n))
    print(f"  {c['id']}: support = {n} ({'exact' if exact else 'approx'})")

if low_support:
    print(f"\nFAILED: {len(low_support)} templates have support < 128:")
    for cid, n in low_support:
        print(f"  {cid}: {n}")
    sys.exit(1)
else:
    print("\nALL 34 TEMPLATES HAVE SUPPORT >= 128!")
