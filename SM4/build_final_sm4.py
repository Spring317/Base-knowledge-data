#!/usr/bin/env python3
"""
Apply ALL SM4_review.md corrections to composite.jsonl and graphs.jsonl.

Fixes applied:
  Issue 1: Rewrite templates as single questions (data + one request)
  Issue 2: Fix question/graph value mismatch (c30, c34)
  Issue 3a: Derivative formula giveaway → ask "find derivative of arcsin/arctan"
  Issue 3b: Volume pseudo-atom → present rotation context
  Issue 3c: Solve pseudo-addition → present IVP context
  Issue 4a: c31 y0=0 no-op → y0 ≥ 1
  Issue 4b: iv=0 no-op → iv ≥ 1
  Issue 4c: arctan→sec² cancellation → replace atom pairs in c05,c10,c16,c18,c29
"""
import json, sys, random
from pathlib import Path
from fractions import Fraction

HERE = Path(__file__).resolve().parent
SM4  = Path("/mnt/storage/Base-knowledge-data/SM4")
ROOT = SM4.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SM4))

import generate, program, kernel, check_program
import atoms as sm4_atoms

random.seed(42)

# Helper lambdas (same as build_final_sm4.py)
Q = lambda v: {"question": v}
L = lambda v: {"literal": v}
R = lambda n: {"ref": n}

composites = []
graphs = []

def add(cid, atoms, template, vars_, nodes, ret, derive=None, constraints=None, ex_vars=None):
    """Add a composite+graph pair, auto-computing the worked example."""
    comp = {"id": cid, "unit": "SM4", "atoms": atoms, "template": template, "vars": vars_}
    if derive:
        comp["derive"] = derive
    if constraints:
        comp["constraints"] = constraints

    g = {"id": cid, "unit": "SM4",
         "nodes": nodes,
         "return": ret if isinstance(ret, dict) else {"ref": ret}}

    # Compute example answer
    v = dict(ex_vars)
    for name in generate.derive_order(comp.get("derive") or {}):
        v[name] = generate.evaluate(comp["derive"][name], v)
    ret_ref = ret["ref"] if isinstance(ret, dict) else ret
    ans, _ = program.run([n for n in nodes], v, ret_ref)
    shown = str(kernel.display(ans, comp.get("display"))).strip()
    comp["example"] = {"vars": ex_vars, "answer": shown}

    composites.append(comp)
    graphs.append(g)


# =====================================================================
#                      DEPTH 2  (10 composites)
# =====================================================================

# c01: partial_frac → power
add("c01_partfrac_power",
    ["spec.integ.partial_frac", "spec.integ.power"],
    "Let M/(x − {p}) + N/(x − {q}) = {A}/((x − {p})(x − {q})). Evaluate ∫₀^|M| x^{n} dx.",
    {"A": {"type":"int","min":1,"max":6},
     "p": {"type":"int","min":-5,"max":0},
     "q": {"type":"int","min":1,"max":6},
     "n": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.integ.partial_frac",
      "args":{"A":Q("A"),"p":Q("p"),"q":Q("q")}},
     {"node_id":"k1","atom_id":"kernel.get_entry","args":{"v":R("n1"),"i":L(0)}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("k1")}},
     {"node_id":"n2","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k2")}}],
    {"ref":"n2"},
    ex_vars={"A":1,"p":-1,"q":1,"n":2})

# c02: implicit_grad → exp_linear
add("c02_implicit_exp",
    ["spec.deriv.implicit_grad", "spec.integ.exp_linear"],
    "On the implicit curve F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let m = |dy/dx|. Evaluate ∫₀^m e^x dx.",
    {"Fx": {"type":"int","min":-8,"max":8,"exclude":[0]},
     "Fy": {"type":"int","min":1,"max":10}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n1")}},
     {"node_id":"n2","atom_id":"spec.integ.exp_linear",
      "args":{"k":L(1),"a":L(0),"b":R("k1")}}],
    {"ref":"n2"},
    ex_vars={"Fx":2,"Fy":1})

# c03: margin_of_error → exp_linear
add("c03_margin_exp",
    ["spec.stats.margin_of_error", "spec.integ.exp_linear"],
    "The margin of error for a confidence interval with z = {z}, s = {s}, n = {n_val} is E. Evaluate ∫₀^E e^({k}x) dx.",
    {"z": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":8},
     "n_val": {"type":"choice","values":[4,9,16,25,36,49,64]},
     "k": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n1")}}],
    {"ref":"n2"},
    ex_vars={"z":1,"s":2,"n_val":4,"k":1})

# c04: arcsin_eval → sin_sq
add("c04_arcsin_sinsq",
    ["spec.trig.arcsin_eval", "spec.integ.sin_sq"],
    "Let θ = arcsin({x_disp}). Evaluate {C}·∫_{a_disp}^θ sin²(x) dx.",
    {"idx_x": {"type":"int","min":0,"max":4},
     "idx_a": {"type":"int","min":0,"max":4},
     "C": {"type":"int","min":1,"max":10}},
    [{"node_id":"n1","atom_id":"spec.trig.arcsin_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":Q("a_val"),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n2")}}],
    {"ref":"k1"},
    derive={"x_val":"[Fraction(-1,1),Fraction(-1,2),Fraction(0,1),Fraction(1,2),Fraction(1,1)][idx_x]",
            "x_disp":"['-1','-1/2','0','1/2','1'][idx_x]",
            "a_val":"[-pi/2,-pi/6,0,pi/6,pi/2][idx_a]",
            "a_disp":"['-π/2','-π/6','0','π/6','π/2'][idx_a]"},
    ex_vars={"idx_x":3,"idx_a":2,"C":1})

# c05: arctan_eval → cos_sq  [REDESIGNED: was sec_sq which cancels with arctan]
add("c05_arctan_cossq",
    ["spec.trig.arctan_eval", "spec.integ.cos_sq"],
    "Let θ = arctan({x_disp}). Evaluate {C}·∫₀^θ cos²(x) dx.",
    {"idx": {"type":"int","min":0,"max":15},
     "C": {"type":"int","min":1,"max":12}},
    [{"node_id":"n1","atom_id":"spec.trig.arctan_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n2")}}],
    {"ref":"k1"},
    derive={"x_val":"[-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10][idx]",
            "x_disp":"['-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10'][idx]"},
    ex_vars={"idx":6,"C":1})

# c06: arccos_eval → cos_sq
add("c06_arccos_cossq",
    ["spec.trig.arccos_eval", "spec.integ.cos_sq"],
    "Let θ = arccos({x_disp}). Evaluate {C}·∫_{a_disp}^θ cos²(x) dx.",
    {"idx_x": {"type":"int","min":0,"max":4},
     "idx_a": {"type":"int","min":0,"max":4},
     "C": {"type":"int","min":1,"max":10}},
    [{"node_id":"n1","atom_id":"spec.trig.arccos_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":Q("a_val"),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n2")}}],
    {"ref":"k1"},
    derive={"x_val":"[Fraction(-1,1),Fraction(-1,2),Fraction(0,1),Fraction(1,2),Fraction(1,1)][idx_x]",
            "x_disp":"['-1','-1/2','0','1/2','1'][idx_x]",
            "a_val":"[0,pi/6,pi/4,pi/3,pi/2][idx_a]",
            "a_disp":"['0','π/6','π/4','π/3','π/2'][idx_a]"},
    ex_vars={"idx_x":3,"idx_a":0,"C":1})

# c07: margin_of_error → reciprocal
add("c07_margin_recip",
    ["spec.stats.margin_of_error", "spec.integ.reciprocal"],
    "The margin of error for z = {z}, s = {s}, n = {n_val} is E. Evaluate ∫₁^E (1/x) dx.",
    {"z": {"type":"int","min":2,"max":5},
     "s": {"type":"int","min":2,"max":10},
     "n_val": {"type":"choice","values":[4,9,16,25,36,49]}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.integ.reciprocal",
      "args":{"a":L(1),"b":R("n1")}}],
    {"ref":"n2"},
    constraints=["z * s > int(sqrt(n_val))"],
    ex_vars={"z":2,"s":4,"n_val":4})

# c08: implicit_grad → power
add("c08_implicit_power",
    ["spec.deriv.implicit_grad", "spec.integ.power"],
    "On F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let m = |dy/dx|. Evaluate ∫₀^m x^{n} dx.",
    {"Fx": {"type":"int","min":1,"max":8},
     "Fy": {"type":"int","min":1,"max":6},
     "n": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n1")}},
     {"node_id":"n2","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k1")}}],
    {"ref":"n2"},
    ex_vars={"Fx":2,"Fy":1,"n":2})

# c09: arcsin_form → sin_sq
add("c09_arcsinform_sinsq",
    ["spec.integ.arcsin_form", "spec.integ.sin_sq"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx. Evaluate {C}·∫₀^θ sin²(x) dx.",
    {"c": {"type":"int","min":3,"max":12},
     "b": {"type":"int","min":1,"max":11},
     "C": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n2")}}],
    {"ref":"k1"},
    derive={"csq":"c * c"},
    constraints=["b < c"],
    ex_vars={"c":4,"b":2,"C":1})

# c10: arctan_form → sin_sq  [REDESIGNED: was sec_sq which cancels]
add("c10_arctanform_sinsq",
    ["spec.integ.arctan_form", "spec.integ.sin_sq"],
    "Let θ = ∫₀^{b} {c}/({csq} + x²) dx. Evaluate {C}·∫₀^θ sin²(x) dx.",
    {"c": {"type":"int","min":1,"max":8},
     "b": {"type":"int","min":1,"max":15},
     "C": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.integ.arctan_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n2")}}],
    {"ref":"k1"},
    derive={"csq":"c * c"},
    ex_vars={"c":1,"b":1,"C":1})


# =====================================================================
#                      DEPTH 3  (8 composites)
# =====================================================================

# c11: implicit_grad → related_rate → arctan_val  [FIX 3a: no formula giveaway]
add("c11_implicit_rate_arctanval",
    ["spec.deriv.implicit_grad", "spec.deriv.related_rate", "spec.deriv.arctan_val"],
    "On F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let v = dy/dt when dx/dt = {dxdt_d}. Find the derivative of arctan(x) evaluated at x = v.",
    {"Fx": {"type":"int","min":-6,"max":6},
     "Fy": {"type":"int","min":1,"max":6},
     "r": {"type":"int","min":-4,"max":4,"exclude":[0]},
     "s": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"n2","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n1"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}}],
    {"ref":"n3"},
    derive={"dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    ex_vars={"Fx":2,"Fy":1,"r":1,"s":1})

# c12: margin → arctan_val → exp  [FIX 3a]
add("c12_margin_arctanval_exp",
    ["spec.stats.margin_of_error", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "The margin of error for z = {z}, s = {s}, n = {n_val} is E. Let u be the derivative of arctan(x) at x = E. Evaluate ∫₀^u e^({k}x) dx.",
    {"z": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":6},
     "n_val": {"type":"choice","values":[4,9,16,25,36,49,64]},
     "k": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.deriv.arctan_val","args":{"x":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n2")}}],
    {"ref":"n3"},
    ex_vars={"z":1,"s":2,"n_val":4,"k":1})

# c13: margin → arcsin_val → arctan_val  [FIX 3a]
add("c13_margin_arcsinval_arctanval",
    ["spec.stats.margin_of_error", "spec.deriv.arcsin_val", "spec.deriv.arctan_val"],
    "The margin of error for z = {z}, s = {s}, n = {n_val} is E. Let v be the derivative of arcsin(x) at x = E. Find {C} times the derivative of arctan(x) at x = v.",
    {"z": {"type":"int","min":1,"max":2},
     "s": {"type":"int","min":1,"max":4},
     "n_val": {"type":"choice","values":[25,36,49,64,81,100,144]},
     "C": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.deriv.arcsin_val","args":{"x":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n3")}}],
    {"ref":"k1"},
    constraints=["z * s < int(sqrt(n_val))"],
    ex_vars={"z":1,"s":1,"n_val":25,"C":1})

# c14: arcsin_eval → cos_sq → shm_eval
add("c14_arcsin_cossq_shm",
    ["spec.trig.arcsin_eval", "spec.integ.cos_sq", "spec.dynamics.shm_eval"],
    "Let θ = arcsin({x_disp}) and I = ∫₀^θ cos²(x) dx. A particle in SHM has x(t) = {A} cos({omega_val}t). Find x(I).",
    {"idx": {"type":"int","min":0,"max":3},
     "A": {"type":"int","min":1,"max":8},
     "omega_val": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.trig.arcsin_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":Q("omega_val"),"phi":L(0),"t":R("n2")}}],
    {"ref":"n3"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1),Fraction(-1,2)][idx]",
            "x_disp":"['0','1/2','1','-1/2'][idx]"},
    ex_vars={"idx":1,"A":2,"omega_val":1})

# c15: arccos_eval → sin_sq → reciprocal
add("c15_arccos_sinsq_recip",
    ["spec.trig.arccos_eval", "spec.integ.sin_sq", "spec.integ.reciprocal"],
    "Let θ = arccos({x_disp}) and I = ∫₀^θ sin²(x) dx. Evaluate {C}·∫₁^({shift} + I) (1/x) dx.",
    {"idx": {"type":"int","min":0,"max":3},
     "C": {"type":"int","min":1,"max":10},
     "shift": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.trig.arccos_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.add","args":{"x":Q("shift"),"y":R("n2")}},
     {"node_id":"n3","atom_id":"spec.integ.reciprocal","args":{"a":L(1),"b":R("k1")}},
     {"node_id":"k2","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n3")}}],
    {"ref":"k2"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1),Fraction(-1,2)][idx]",
            "x_disp":"['0','1/2','1','-1/2'][idx]"},
    ex_vars={"idx":0,"C":1,"shift":1})

# c16: margin → sec_sq → reciprocal  [REDESIGNED: was arctan→sec_sq which cancels]
add("c16_margin_secsq_recip",
    ["spec.stats.margin_of_error", "spec.integ.sec_sq", "spec.integ.reciprocal"],
    "The margin of error for z = {z}, s = {s}, n = {n_val} is E. Let T = ∫₀^E sec²(x) dx. Evaluate {C}·∫₁^(1 + T) (1/x) dx.",
    {"z": {"type":"int","min":1,"max":1},
     "s": {"type":"int","min":1,"max":3},
     "n_val": {"type":"choice","values":[4,9,16,25,36,49]},
     "C": {"type":"int","min":1,"max":10}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.integ.sec_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"k1","atom_id":"kernel.add","args":{"x":L(1),"y":R("n2")}},
     {"node_id":"n3","atom_id":"spec.integ.reciprocal","args":{"a":L(1),"b":R("k1")}},
     {"node_id":"k2","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n3")}}],
    {"ref":"k2"},
    ex_vars={"z":1,"s":1,"n_val":4,"C":1})

# c17: arcsin_form → cos_sq → arctan_val  [FIX 3a]
add("c17_arcsinform_cossq_arctanval",
    ["spec.integ.arcsin_form", "spec.integ.cos_sq", "spec.deriv.arctan_val"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx and I = ∫₀^θ cos²(x) dx. Find {C} times the derivative of arctan(x) at x = I.",
    {"c": {"type":"int","min":3,"max":10},
     "b": {"type":"int","min":1,"max":9},
     "C": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n3")}}],
    {"ref":"k1"},
    derive={"csq":"c * c"},
    constraints=["b < c"],
    ex_vars={"c":4,"b":2,"C":1})

# c18: arcsin_form → sec_sq → exp  [REDESIGNED: was arctan_form→sec_sq which cancels]
add("c18_arcsinform_secsq_exp",
    ["spec.integ.arcsin_form", "spec.integ.sec_sq", "spec.integ.exp_linear"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx and T = ∫₀^θ sec²(x) dx. Evaluate ∫₀^T e^({k}x) dx.",
    {"c": {"type":"int","min":2,"max":9},
     "b": {"type":"int","min":1,"max":8},
     "k": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.integ.sec_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n2")}}],
    {"ref":"n3"},
    derive={"csq":"c * c"},
    constraints=["b < c"],
    ex_vars={"c":3,"b":1,"k":1})


# =====================================================================
#                      DEPTH 4  (7 composites)
# =====================================================================

# c19: arcsin_eval → shm_eval → arctan_val → related_rate  [FIX 3a]
add("c19_arcsin_shm_arctanval_rate",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate"],
    "Let θ = arcsin({x_disp}) and x₀ = {A} cos(θ). Let m be the derivative of arctan(x) at x = x₀. If dx/dt = {dxdt_d}, find m·(dx/dt).",
    {"idx": {"type":"int","min":0,"max":3},
     "A": {"type":"int","min":1,"max":5},
     "r": {"type":"int","min":1,"max":5},
     "s": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.trig.arcsin_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"n4","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n3"),"dx_dt":Q("dxdt_val")}}],
    {"ref":"n4"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1),Fraction(-1,2)][idx]",
            "x_disp":"['0','1/2','1','-1/2'][idx]",
            "dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    ex_vars={"idx":1,"A":2,"r":3,"s":1})

# c20: arctan_form → shm_eval → arctan_val → exp  [FIX 3a]
add("c20_arctanform_shm_arctanval_exp",
    ["spec.integ.arctan_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "Let θ = ∫₀^{b} {c}/({csq} + x²) dx and x₀ = {A} cos(θ). Let m be the derivative of arctan(x) at x = x₀. Evaluate ∫₀^m e^({k}x) dx.",
    {"c": {"type":"int","min":1,"max":5},
     "b": {"type":"int","min":1,"max":6},
     "A": {"type":"int","min":1,"max":4},
     "k": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.integ.arctan_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"n4","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n3")}}],
    {"ref":"n4"},
    derive={"csq":"c * c"},
    ex_vars={"c":1,"b":1,"A":1,"k":1})

# c21: partial_frac → power → by_parts → volume  [FIX 3b,4b: iv≥1, volume context]
add("c21_partfrac_power_parts_vol",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Let M/(x − {p}) + N/(x − {q}) = {A}/((x − {p})(x − {q})), and b = |M|. Let I = ∫₀^b x^{n} dx. In integration by parts, [uv]₀^a = I and ∫₀^a v du = {iv}; let J = [uv]₀^a − ∫₀^a v du. A curve y = f(x) is rotated about the x-axis with ∫[f(x)]² dx = J. Find the volume of the solid.",
    {"A": {"type":"int","min":1,"max":5},
     "p": {"type":"int","min":-4,"max":0},
     "q": {"type":"int","min":1,"max":5},
     "n": {"type":"int","min":1,"max":3},
     "iv": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.integ.partial_frac",
      "args":{"A":Q("A"),"p":Q("p"),"q":Q("q")}},
     {"node_id":"k1","atom_id":"kernel.get_entry","args":{"v":R("n1"),"i":L(0)}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("k1")}},
     {"node_id":"n2","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k2")}},
     {"node_id":"n3","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n2"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n4","atom_id":"spec.integ.volume_disk",
      "args":{"integral_f_sq":R("n3")}}],
    {"ref":"n4"},
    ex_vars={"A":1,"p":-1,"q":1,"n":2,"iv":1})

# c22: partial_frac → reciprocal → exp → solve  [FIX 3c: IVP context]
add("c22_partfrac_recip_exp_solve",
    ["spec.integ.partial_frac", "spec.integ.reciprocal", "spec.integ.exp_linear", "spec.diffeq.solve_direct"],
    "Let M/(x − {p}) + N/(x − {q}) = {A}/((x − {p})(x − {q})), and b = |M| + 1. Let I = ∫₁^b (1/x) dx and J = ∫₀^I e^x dx. Given dy/dx = f(x), y(0) = {y0}, and ∫₀^a f(x) dx = J, find y(a).",
    {"A": {"type":"int","min":1,"max":5},
     "p": {"type":"int","min":-4,"max":0},
     "q": {"type":"int","min":1,"max":5},
     "y0": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.integ.partial_frac",
      "args":{"A":Q("A"),"p":Q("p"),"q":Q("q")}},
     {"node_id":"k1","atom_id":"kernel.get_entry","args":{"v":R("n1"),"i":L(0)}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("k1")}},
     {"node_id":"k3","atom_id":"kernel.add","args":{"x":L(1),"y":R("k2")}},
     {"node_id":"n2","atom_id":"spec.integ.reciprocal","args":{"a":L(1),"b":R("k3")}},
     {"node_id":"n3","atom_id":"spec.integ.exp_linear","args":{"k":L(1),"a":L(0),"b":R("n2")}},
     {"node_id":"n4","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n3")}}],
    {"ref":"n4"},
    ex_vars={"A":1,"p":-1,"q":1,"y0":2})

# c23: partial_frac → power → reciprocal → exp
add("c23_partfrac_power_recip_exp",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.reciprocal", "spec.integ.exp_linear"],
    "In the partial fractions of {A}/((x − {p})(x − {q})), let M be the coefficient of 1/(x − {p}). With b = |M| + 1, I = ∫₀^b x dx, and J = ∫₁^(1+I) (1/x) dx, evaluate ∫₀^J e^({k}x) dx.",
    {"A": {"type":"int","min":1,"max":6},
     "p": {"type":"int","min":-5,"max":0},
     "q": {"type":"int","min":1,"max":6},
     "k": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.integ.partial_frac",
      "args":{"A":Q("A"),"p":Q("p"),"q":Q("q")}},
     {"node_id":"k1","atom_id":"kernel.get_entry","args":{"v":R("n1"),"i":L(0)}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("k1")}},
     {"node_id":"k3","atom_id":"kernel.add","args":{"x":L(1),"y":R("k2")}},
     {"node_id":"n2","atom_id":"spec.integ.power","args":{"n":L(1),"a":L(0),"b":R("k3")}},
     {"node_id":"k4","atom_id":"kernel.add","args":{"x":L(1),"y":R("n2")}},
     {"node_id":"n3","atom_id":"spec.integ.reciprocal","args":{"a":L(1),"b":R("k4")}},
     {"node_id":"n4","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n3")}}],
    {"ref":"n4"},
    ex_vars={"A":1,"p":-1,"q":1,"k":1})

# c24: arcsin_eval → sin_sq → by_parts → volume  [FIX 3b,4b]
add("c24_arcsin_sinsq_parts_vol",
    ["spec.trig.arcsin_eval", "spec.integ.sin_sq", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Let θ = arcsin({x_disp}) and I = ∫₀^θ sin²(x) dx. In integration by parts, [uv]₀^a = I and ∫₀^a v du = {iv}; let J = [uv]₀^a − ∫₀^a v du. A curve is rotated about the x-axis with ∫[f(x)]² dx = {C}·J. Find the volume.",
    {"idx": {"type":"int","min":0,"max":3},
     "iv": {"type":"int","min":1,"max":5},
     "C": {"type":"int","min":1,"max":8}},
    [{"node_id":"n1","atom_id":"spec.trig.arcsin_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n2"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":Q("C"),"y":R("n3")}},
     {"node_id":"n4","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("k1")}}],
    {"ref":"n4"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1),Fraction(-1,2)][idx]",
            "x_disp":"['0','1/2','1','-1/2'][idx]"},
    ex_vars={"idx":1,"iv":1,"C":1})

# c25: arccos_eval → cos_sq → by_parts → arcsin_val  [FIX 3a,4b]
add("c25_arccos_cossq_parts_arcsinval",
    ["spec.trig.arccos_eval", "spec.integ.cos_sq", "spec.integ.by_parts_step", "spec.deriv.arcsin_val"],
    "Let θ = arccos({x_disp}) and I = ∫₀^θ cos²(x) dx. With [uv]₀^a = I and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du and u = J/{scale}. Find the derivative of arcsin(x) at x = u.",
    {"idx": {"type":"int","min":0,"max":2},
     "iv": {"type":"int","min":1,"max":4},
     "scale": {"type":"int","min":10,"max":26}},
    [{"node_id":"n1","atom_id":"spec.trig.arccos_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n2"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"k1","atom_id":"kernel.divide","args":{"x":R("n3"),"y":Q("scale")}},
     {"node_id":"n4","atom_id":"spec.deriv.arcsin_val","args":{"x":R("k1")}}],
    {"ref":"n4"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1)][idx]",
            "x_disp":"['0','1/2','1'][idx]"},
    ex_vars={"idx":0,"iv":1,"scale":10})


# =====================================================================
#                      DEPTH 5  (5 composites)
# =====================================================================

# c26: arcsin_form → shm → arctan_val → rate → solve  [FIX 3a,3c]
add("c26_arcsinform_shm_arctanval_rate_solve",
    ["spec.integ.arcsin_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx and x₀ = {A} cos(θ). Let m be the derivative of arctan(x) at x = x₀. If dx/dt = {dxdt_d}, let v = m·(dx/dt). Given dz/dt = v, z(0) = {z0}, find z({T}).",
    {"c": {"type":"int","min":2,"max":6},
     "b": {"type":"int","min":1,"max":5},
     "A": {"type":"int","min":1,"max":3},
     "r": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":2},
     "z0": {"type":"int","min":1,"max":4},
     "T": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"n4","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n3"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":R("n4"),"y":Q("T")}},
     {"node_id":"n5","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("z0"),"integral_value":R("k1")}}],
    {"ref":"n5"},
    derive={"csq":"c * c","dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    constraints=["b < c"],
    ex_vars={"c":2,"b":1,"A":1,"r":1,"s":1,"z0":1,"T":2})

# c27: arccos_eval → shm → arctan_val → rate → power  [FIX 3a]
add("c27_arccos_shm_arctanval_rate_power",
    ["spec.trig.arccos_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.integ.power"],
    "Let θ = arccos({x_disp}) and x₀ = {A} cos(θ). Let m be the derivative of arctan(x) at x = x₀. If dx/dt = {dxdt_d}, let v = |m·(dx/dt)|. Evaluate ∫₀^v x^{n} dx.",
    {"idx": {"type":"int","min":0,"max":2},
     "A": {"type":"int","min":1,"max":3},
     "r": {"type":"int","min":1,"max":4},
     "s": {"type":"int","min":1,"max":2},
     "n": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.trig.arccos_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}},
     {"node_id":"n4","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n3"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n4")}},
     {"node_id":"n5","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k1")}}],
    {"ref":"n5"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1)][idx]",
            "x_disp":"['0','1/2','1'][idx]",
            "dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    ex_vars={"idx":0,"A":1,"r":2,"s":1,"n":1})

# c28: partial_frac → power → volume → by_parts → solve  [FIX 3b,3c,4b]
add("c28_partfrac_power_vol_parts_solve",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.volume_disk", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "Let M/(x − {p}) + N/(x − {q}) = {A}/((x − {p})(x − {q})), and b = |M|. Let I = ∫₀^b x^{n} dx. A curve is rotated about the x-axis with ∫[f(x)]² dx = I; let V be the volume. With [uv]₀^a = V and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du. Given dy/dx = g(x), y(0) = {y0}, ∫g dx = J, find y.",
    {"A": {"type":"int","min":1,"max":5},
     "p": {"type":"int","min":-4,"max":0},
     "q": {"type":"int","min":1,"max":4},
     "n": {"type":"int","min":1,"max":3},
     "iv": {"type":"int","min":1,"max":3},
     "y0": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.integ.partial_frac",
      "args":{"A":Q("A"),"p":Q("p"),"q":Q("q")}},
     {"node_id":"k1","atom_id":"kernel.get_entry","args":{"v":R("n1"),"i":L(0)}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("k1")}},
     {"node_id":"n2","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k2")}},
     {"node_id":"n3","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n2")}},
     {"node_id":"n4","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n3"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n5","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n4")}}],
    {"ref":"n5"},
    ex_vars={"A":1,"p":-1,"q":1,"n":2,"iv":1,"y0":1})

# c29: arctan_eval → sin_sq → by_parts → exp → solve  [REDESIGNED to avoid sec² cancel; FIX 3c,4b]
add("c29_arctan_sinsq_parts_exp_solve",
    ["spec.trig.arctan_eval", "spec.integ.sin_sq", "spec.integ.by_parts_step", "spec.integ.exp_linear", "spec.diffeq.solve_direct"],
    "Let θ = arctan({x_disp}) and I = ∫₀^θ sin²(x) dx. With [uv]₀^a = I and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du. Let K = ∫₀^J e^({k}x) dx. Given dy/dx = e^({k}x), y(0) = {y0}, find y(J).",
    {"idx": {"type":"int","min":0,"max":5},
     "iv": {"type":"int","min":1,"max":4},
     "k": {"type":"int","min":1,"max":3},
     "y0": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.trig.arctan_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n2"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n4","atom_id":"spec.integ.exp_linear",
      "args":{"k":Q("k"),"a":L(0),"b":R("n3")}},
     {"node_id":"n5","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n4")}}],
    {"ref":"n5"},
    derive={"x_val":"[1,2,3,4,5,6][idx]",
            "x_disp":"['1','2','3','4','5','6'][idx]"},
    ex_vars={"idx":0,"iv":1,"k":1,"y0":1})

# c30: implicit → arcsin_val → rate → by_parts → volume  [FIX 2,3a,3b,4b]
add("c30_implicit_arcsinval_rate_parts_vol",
    ["spec.deriv.implicit_grad", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "On F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let u = dy/dx. Let w be the derivative of arcsin(x) at x = u. If dx/dt = {dxdt_d}, let v = |w·(dx/dt)|. With [uv]₀^a = v and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du. A curve is rotated about the x-axis with ∫[f(x)]² dx = J. Find the volume.",
    {"Fx": {"type":"int","min":1,"max":4},
     "Fy": {"type":"int","min":5,"max":10},
     "r": {"type":"int","min":1,"max":4},
     "s": {"type":"int","min":1,"max":3},
     "iv": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"n2","atom_id":"spec.deriv.arcsin_val","args":{"x":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n2"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n3")}},
     {"node_id":"n4","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("k1"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n5","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n4")}}],
    {"ref":"n5"},
    derive={"dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    ex_vars={"Fx":3,"Fy":5,"r":1,"s":1,"iv":1})


# =====================================================================
#                      DEPTH 6  (4 composites)
# =====================================================================

# c31: margin → shm → arcsin_val → rate → solve → volume  [FIX 2,3a,3b,4a]
add("c31_margin_shm_arcsinval_rate_solve_vol",
    ["spec.stats.margin_of_error", "spec.dynamics.shm_eval", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.volume_disk"],
    "The margin of error for z = {z}, s = {s}, n = {n_val} is E. A particle has x(t) = {A_disp} cos(t); let x₀ = x(E). Let w be the derivative of arcsin(x) at x = x₀. If dx/dt = {dxdt_d}, let v = w·(dx/dt). Given dw/dt = v, w(0) = {w0}, find w({T}). A curve is rotated about the x-axis with ∫[f(x)]² dx = w({T}). Find the volume.",
    {"z": {"type":"int","min":1,"max":2},
     "s": {"type":"int","min":1,"max":3},
     "n_val": {"type":"choice","values":[16,25,36,49,64]},
     "A_denom": {"type":"int","min":2,"max":5},
     "r": {"type":"int","min":1,"max":3},
     "s_rate": {"type":"int","min":1,"max":2},
     "T": {"type":"int","min":1,"max":3},
     "w0": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A_val"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arcsin_val","args":{"x":R("n2")}},
     {"node_id":"n4","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n3"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.multiply","args":{"x":R("n4"),"y":Q("T")}},
     {"node_id":"n5","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("w0"),"integral_value":R("k1")}},
     {"node_id":"n6","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n5")}}],
    {"ref":"n6"},
    derive={"A_val":"Fraction(1, A_denom)","A_disp":"f'1/{A_denom}'",
            "dxdt_val":"Fraction(r, s_rate)","dxdt_d":"str(Fraction(r, s_rate))"},
    ex_vars={"z":1,"s":1,"n_val":16,"A_denom":2,"r":1,"s_rate":1,"T":1,"w0":1})

# c32: arcsin_eval → shm → arcsin_form → sin_sq → volume → solve  [FIX 3b,3c]
add("c32_arcsin_shm_arcsinform_sinsq_vol_solve",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.integ.arcsin_form", "spec.integ.sin_sq", "spec.integ.volume_disk", "spec.diffeq.solve_direct"],
    "Let θ₁ = arcsin({x_disp}) and x₀ = {A} cos(θ₁). Let θ₂ = ∫₀^x₀ 1/√({csq} − x²) dx and I = ∫₀^θ₂ sin²(x) dx. A curve is rotated about the x-axis with ∫[f(x)]² dx = I; let V be the volume. Given dy/dx = g(x), y(0) = {y0}, ∫g dx = V, find y.",
    {"idx": {"type":"int","min":0,"max":2},
     "A": {"type":"int","min":1,"max":3},
     "c": {"type":"int","min":4,"max":7},
     "y0": {"type":"int","min":1,"max":6}},
    [{"node_id":"n1","atom_id":"spec.trig.arcsin_eval","args":{"x":Q("x_val")}},
     {"node_id":"n2","atom_id":"spec.dynamics.shm_eval",
      "args":{"A":Q("A"),"omega":L(1),"phi":L(0),"t":R("n1")}},
     {"node_id":"n3","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":R("n2")}},
     {"node_id":"n4","atom_id":"spec.integ.sin_sq","args":{"a":L(0),"b":R("n3")}},
     {"node_id":"n5","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n4")}},
     {"node_id":"n6","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n5")}}],
    {"ref":"n6"},
    derive={"x_val":"[0,Fraction(1,2),Fraction(1,1)][idx]",
            "x_disp":"['0','1/2','1'][idx]",
            "csq":"c * c"},
    constraints=["A < c"],
    ex_vars={"idx":0,"A":1,"c":4,"y0":2})

# c33: arcsin_val → rate → solve → volume → by_parts → solve  [FIX 3a,3b,3c,4b]
add("c33_arcsinval_rate_solve_vol_parts_solve",
    ["spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.volume_disk", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "At x = {xd}, let m be the derivative of arcsin(x). If dx/dt = {dxdt_d}, let v = |m·(dx/dt)|. Given dw/dt = v, w(0) = {w0}, find w(1). A curve is rotated about the x-axis with ∫[f(x)]² dx = w(1); let V be the volume. With [uv]₀^a = V and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du. Given dy/dx = h(x), y(0) = {y0}, ∫h dx = J, find y.",
    {"p": {"type":"int","min":1,"max":4},
     "q": {"type":"int","min":5,"max":10},
     "r": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":2},
     "w0": {"type":"int","min":1,"max":3},
     "y0": {"type":"int","min":1,"max":4},
     "iv": {"type":"int","min":1,"max":3}},
    [{"node_id":"n1","atom_id":"spec.deriv.arcsin_val","args":{"x":Q("xv")}},
     {"node_id":"n2","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n1"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n2")}},
     {"node_id":"n3","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("w0"),"integral_value":R("k1")}},
     {"node_id":"n4","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n3")}},
     {"node_id":"n5","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n4"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n6","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n5")}}],
    {"ref":"n6"},
    derive={"xv":"Fraction(p, q)","xd":"str(Fraction(p, q))",
            "dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    constraints=["p < q"],
    ex_vars={"p":3,"q":5,"r":1,"s":1,"w0":1,"y0":1,"iv":1})

# c34: implicit → arcsin_val → rate → solve → by_parts → solve  [FIX 2,3a,3c,4b]
add("c34_implicit_arcsinval_rate_solve_parts_solve",
    ["spec.deriv.implicit_grad", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "On F(x, y) = 0 with ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let u = dy/dx. Let w be the derivative of arcsin(x) at x = u. If dx/dt = {dxdt_d}, let v = |w·(dx/dt)|. Given dξ/dt = v, ξ(0) = {w0}, find ξ(1). With [uv]₀^a = ξ(1) and ∫₀^a v du = {iv}, let J = [uv]₀^a − ∫₀^a v du. Given dy/dx = h(x), y(0) = {y0}, ∫h dx = J, find y.",
    {"Fx": {"type":"int","min":1,"max":3},
     "Fy": {"type":"int","min":4,"max":8},
     "r": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":2},
     "w0": {"type":"int","min":1,"max":3},
     "iv": {"type":"int","min":1,"max":3},
     "y0": {"type":"int","min":1,"max":4}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"n2","atom_id":"spec.deriv.arcsin_val","args":{"x":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n2"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n3")}},
     {"node_id":"n4","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("w0"),"integral_value":R("k1")}},
     {"node_id":"n5","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n4"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n6","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n5")}}],
    {"ref":"n6"},
    derive={"dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    constraints=["Fx < Fy"],
    ex_vars={"Fx":1,"Fy":4,"r":1,"s":1,"w0":1,"iv":1,"y0":1})


# =====================================================================
#                      WRITE OUTPUT & VERIFY
# =====================================================================

print(f"Generated {len(composites)} composites and {len(graphs)} graphs")

# Check depth distribution
kernel_only = {a for a, kind in kernel.KIND.items() if kind == "kernel"}
depth_counts = {}
for g in graphs:
    d = check_program.knowledge_depth(g["nodes"], (g["return"]["ref"] if isinstance(g["return"], dict) else g["return"]))
    depth_counts[d] = depth_counts.get(d, 0) + 1
    print(f"  {g['id']:<50} depth {d}")
print(f"\nDepth distribution: {dict(sorted(depth_counts.items()))}")
target = {2:10, 3:8, 4:7, 5:5, 6:4}
assert depth_counts == target, f"Depth mismatch! Got {depth_counts}, want {target}"
print("✓ Depth distribution matches target")

# Write files
with open(SM4 / "composite.jsonl", "w") as f:
    for c in composites:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"Written {len(composites)} composites to {SM4 / 'composite.jsonl'}")

with open(SM4 / "graphs.jsonl", "w") as f:
    for g in graphs:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
print(f"Written {len(graphs)} graphs to {SM4 / 'graphs.jsonl'}")

# Verify examples
print("\nVerifying examples against check_program...")
fails = 0
for c, g in zip(composites, graphs):
    results = check_program.check(c, g)
    bad = [r for r in results if not r[1]]
    if bad:
        fails += 1
        print(f"  FAIL {c['id']}:")
        for name, ok, detail in bad:
            print(f"    {name}: {detail}")
    else:
        print(f"  {c['id']}: OK")

if fails:
    print(f"\n{fails} composites FAILED check_program!")
    sys.exit(1)
else:
    print(f"\n✓ All {len(composites)} composites passed check_program!")
