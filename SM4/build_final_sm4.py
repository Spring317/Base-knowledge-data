#!/usr/bin/env python3
"""
Master Builder for SM4:
Applies ALL requirements from user review:
1. Atomic templates volume_frac & volume_int: give a function, do NOT give away volume formula.
2. Atomic templates by_parts_compute & by_parts_boundary: do NOT list pre-multiplied pieces or formula.
3. Atomic templates solve_direct_ic & solve_direct_interpret: formulate as real differential equations.
4. Composite templates c01-c10: remove 'Let M/(x-p) + N/(x-q) = ...' formula hints.
5. Composite templates c11-c13, c17, c19, c20: remove explicit 'Find derivative of arctan/arcsin'.
6. Composite templates c21, c24, c25, c28, c29, c30, c31, c33, c34: delete all explicit 'let J = [uv]_0^a - int v du' formulas.
7. Realistic high-depth composites (c26, c27, c31, c32, c33, c34): realistic scenarios, NO duplicate solve_direct.
8. Strictly maintain depth quota {2: 10, 3: 8, 4: 7, 5: 5, 6: 4} and all atom spread in [2, 8].
"""
import json, sys, random, re
from pathlib import Path
from fractions import Fraction

ROOT = Path("/mnt/storage/Base-knowledge-data")
SM4  = ROOT / "SM4"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SM4))

import generate, program, kernel, check_program, validate
import atoms as sm4_atoms

random.seed(42)

# Lambdas for graph node args
Q = lambda v: {"question": v}
L = lambda v: {"literal": v}
R = lambda n: {"ref": n}

# =====================================================================
# PART 1: FIX ATOMIC TEMPLATES
# =====================================================================

with open(SM4 / "templates.jsonl") as f:
    templates = [json.loads(line) for line in f if line.strip()]

for t in templates:
    tid = t["id"]

    # 1. Volume atoms: do not give away formula
    if tid == "volume_frac":
        t["template"] = "Find the exact volume of the solid generated when the region bounded by y = \u221a({k_disp}), the x-axis, and the vertical lines x = 0 and x = 1 is rotated about the x-axis."
        t["vars"] = {
            "p": {"type": "int", "min": 1, "max": 30},
            "q": {"type": "int", "min": 2, "max": 15}
        }
        t["derive"] = {
            "k_val": "Fraction(p, q)",
            "k_disp": "str(Fraction(p, q))"
        }
        t["args"] = {"integral_f_sq": {"question": "k_val"}}

    elif tid == "volume_int":
        t["template"] = "Find the exact volume of the solid formed by rotating the region under the curve y = \u221a({c}) from x = 0 to x = 1 about the x-axis."
        t["vars"] = {
            "c": {"type": "int", "min": 1, "max": 150}
        }
        t["args"] = {"integral_f_sq": {"question": "c"}}

    # 2. Integration by parts: do not list pre-multiplied pieces or formula
    elif tid == "by_parts_compute":
        t["template"] = "Let u(x) and v(x) be differentiable functions with u({b}) = {u_b}, v({b}) = {v_b}, u({a}) = {u_a}, and v({a}) = {v_a}. Given that \u222b_{{a}}^{{b}} v(x) u'(x) dx = {iv}, find the exact value of \u222b_{{a}}^{{b}} u(x) v'(x) dx."
        t["vars"] = {
            "a": {"type": "int", "min": 0, "max": 2},
            "b": {"type": "int", "min": 3, "max": 6},
            "u_b": {"type": "int", "min": 1, "max": 6},
            "v_b": {"type": "int", "min": 1, "max": 6},
            "u_a": {"type": "int", "min": 0, "max": 3},
            "v_a": {"type": "int", "min": 0, "max": 3},
            "iv": {"type": "int", "min": 1, "max": 10}
        }
        t["derive"] = {
            "uv_hi": "u_b * v_b",
            "uv_lo": "u_a * v_a"
        }
        t["constraints"] = ["a < b"]
        t["args"] = {
            "uv_upper": {"question": "uv_hi"},
            "uv_lower": {"question": "uv_lo"},
            "integral_vdu": {"question": "iv"}
        }

    elif tid == "by_parts_boundary":
        t["template"] = "Two differentiable functions f(x) and g(x) satisfy f({b})g({b}) \u2212 f({a})g({a}) = {boundary}. If \u222b_{{a}}^{{b}} g(x) f'(x) dx = {iv}, evaluate \u222b_{{a}}^{{b}} f(x) g'(x) dx."
        t["vars"] = {
            "a": {"type": "int", "min": 0, "max": 2},
            "b": {"type": "int", "min": 3, "max": 6},
            "boundary": {"type": "int", "min": 2, "max": 30},
            "iv": {"type": "int", "min": 1, "max": 15}
        }
        t.pop("constraints", None)
        t.pop("derive", None)
        t["args"] = {
            "uv_upper": {"question": "boundary"},
            "uv_lower": {"literal": 0},
            "integral_vdu": {"question": "iv"}
        }

    # 3. Direct DE solving: real differential equation problems
    elif tid == "solve_direct_ic":
        t["template"] = "Solve the differential equation dy/dx = {k} subject to the initial condition y(0) = {y0}. Find the value of y({x1})."
        t["vars"] = {
            "k": {"type": "int", "min": 1, "max": 10},
            "y0": {"type": "int", "min": 1, "max": 20},
            "x1": {"type": "int", "min": 1, "max": 8}
        }
        t["derive"] = {
            "I_val": "k * x1"
        }
        t.pop("constraints", None)
        t["args"] = {
            "y0": {"question": "y0"},
            "integral_value": {"question": "I_val"}
        }

    elif tid == "solve_direct_interpret":
        t["template"] = "A particle moves in a straight line with constant velocity v(t) = {v} m/s. Its initial position at time t = 0 is s(0) = {s0} m. Find the position of the particle at time t = {t1} s."
        t["vars"] = {
            "v": {"type": "int", "min": 1, "max": 12},
            "s0": {"type": "int", "min": 1, "max": 20},
            "t1": {"type": "int", "min": 1, "max": 8}
        }
        t["derive"] = {
            "disp_val": "v * t1"
        }
        t.pop("constraints", None)
        t["args"] = {
            "y0": {"question": "s0"},
            "integral_value": {"question": "disp_val"}
        }

    # 4. Derivative formula giveaway checks
    elif tid == "deriv_arcsin_formula":
        t["template"] = "Find the slope of the curve y = arcsin(x) at x = {x_disp}."
    elif tid == "deriv_arctan_formula":
        t["template"] = "Find the slope of the curve y = arctan(x) at x = {x_disp}."

with open(SM4 / "templates.jsonl", "w") as f:
    for t in templates:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"✓ Updated {len(templates)} atomic templates in {SM4 / 'templates.jsonl'}")


# =====================================================================
# PART 2: BUILD ALL 34 COMPOSITES AND REFERENCE GRAPHS
# =====================================================================

composites = []
graphs = []

def add(cid, atoms, template, vars_, nodes, ret, derive=None, constraints=None, ex_vars=None):
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


# ---------------------------------------------------------------------
# DEPTH 2 (10 composites)
# ---------------------------------------------------------------------

# c01: partial_frac → power
add("c01_partfrac_power",
    ["spec.integ.partial_frac", "spec.integ.power"],
    "Express {A}/((x − {p})(x − {q})) in partial fractions. Let M be the numerator corresponding to (x − {p}). Evaluate ∫₀^|M| x^{n} dx.",
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
    "A curve is defined implicitly by F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. Let m = |dy/dx|. Evaluate ∫₀^m e^x dx.",
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
    "A sample survey with z = {z}, s = {s}, and sample size n = {n_val} has margin of error E. Evaluate ∫₀^E e^({k}x) dx.",
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
    "Find the value of {C}·∫_{a_disp}^θ sin²(x) dx, where θ = arcsin({x_disp}).",
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

# c05: arctan_eval → cos_sq
add("c05_arctan_cossq",
    ["spec.trig.arctan_eval", "spec.integ.cos_sq"],
    "Find the value of {C}·∫₀^θ cos²(x) dx, where θ = arctan({x_disp}).",
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
    "Find the value of {C}·∫_{a_disp}^θ cos²(x) dx, where θ = arccos({x_disp}).",
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
    "A confidence interval with z = {z}, s = {s}, and n = {n_val} has margin of error E. Evaluate ∫₁^E (1/x) dx.",
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
    "For the implicit curve F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}, let m = |dy/dx|. Evaluate ∫₀^m x^{n} dx.",
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

# c10: arctan_form → sin_sq
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


# ---------------------------------------------------------------------
# DEPTH 3 (8 composites)
# ---------------------------------------------------------------------

# c11: implicit_grad → related_rate → arctan_val
add("c11_implicit_rate_arctanval",
    ["spec.deriv.implicit_grad", "spec.deriv.related_rate", "spec.deriv.arctan_val"],
    "A curve satisfies F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. A particle moves along the curve with dx/dt = {dxdt_d}, giving vertical velocity v = dy/dt. Find the gradient of the curve y = arctan(x) at x = v.",
    {"Fx": {"type":"int","min":-6,"max":6,"exclude":[0]},
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

# c12: margin → arctan_val → exp
add("c12_margin_arctanval_exp",
    ["spec.stats.margin_of_error", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "A sample survey yields margin of error E for z = {z}, s = {s}, and n = {n_val}. Let m be the slope of the curve y = arctan(x) at x = E. Evaluate ∫₀^m e^({k}x) dx.",
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

# c13: margin → arcsin_val → arctan_val
add("c13_margin_arcsinval_arctanval",
    ["spec.stats.margin_of_error", "spec.deriv.arcsin_val", "spec.deriv.arctan_val"],
    "A confidence interval calculated with z = {z}, s = {s}, and n = {n_val} has margin of error E. Let m be the tangent slope of y = arcsin(x) at x = E. Find the tangent slope of the curve y = arctan(x) at x = m.",
    {"z": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":6},
     "n_val": {"type":"choice","values":[25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324]}},
    [{"node_id":"n1","atom_id":"spec.stats.margin_of_error",
      "args":{"z":Q("z"),"s":Q("s"),"n":Q("n_val")}},
     {"node_id":"n2","atom_id":"spec.deriv.arcsin_val","args":{"x":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}}],
    {"ref":"n3"},
    constraints=["z * s < int(sqrt(n_val))"],
    ex_vars={"z":1,"s":1,"n_val":25})

# c14: arcsin_eval → cos_sq → shm_eval
add("c14_arcsin_cossq_shm",
    ["spec.trig.arcsin_eval", "spec.integ.cos_sq", "spec.dynamics.shm_eval"],
    "An oscillator has displacement x(t) = {A} cos({omega_val}t). At time t\u2080 = ∫₀^θ cos²(u) du where θ = arcsin({x_disp}), find the displacement x(t\u2080).",
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

# c16: margin → sec_sq → reciprocal
add("c16_margin_secsq_recip",
    ["spec.stats.margin_of_error", "spec.integ.sec_sq", "spec.integ.reciprocal"],
    "A confidence interval with z = {z}, s = {s}, and n = {n_val} has margin of error E. Let T = ∫₀^E sec²(x) dx. Evaluate {C}·∫₁^(1 + T) (1/x) dx.",
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

# c17: arcsin_form → cos_sq → arctan_val
add("c17_arcsinform_cossq_arctanval",
    ["spec.integ.arcsin_form", "spec.integ.cos_sq", "spec.deriv.arctan_val"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx and I = ∫₀^θ cos²(x) dx. Find the slope of the curve y = arctan(x) at x = I.",
    {"c": {"type":"int","min":3,"max":20},
     "b": {"type":"int","min":1,"max":19}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"n2","atom_id":"spec.integ.cos_sq","args":{"a":L(0),"b":R("n1")}},
     {"node_id":"n3","atom_id":"spec.deriv.arctan_val","args":{"x":R("n2")}}],
    {"ref":"n3"},
    derive={"csq":"c * c"},
    constraints=["b < c"],
    ex_vars={"c":4,"b":2})

# c18: arcsin_form → sec_sq → exp
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


# ---------------------------------------------------------------------
# DEPTH 4 (7 composites)
# ---------------------------------------------------------------------

# c19: arcsin_eval → shm_eval → arctan_val → related_rate
add("c19_arcsin_shm_arctanval_rate",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate"],
    "A tracking camera monitors an object with harmonic position x(t) = {A} cos(t). The camera angle is θ(x) = arctan(x). At time t\u2080 = arcsin({x_disp}), the object is at x\u2080 = x(t\u2080) with velocity dx/dt = {dxdt_d}. Find the camera's angular velocity dθ/dt.",
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

# c20: arctan_form → shm_eval → arctan_val → exp
add("c20_arctanform_shm_arctanval_exp",
    ["spec.integ.arctan_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.integ.exp_linear"],
    "An oscillating particle has position x(t) = {A} cos(t). At time t\u2080 = ∫₀^{b} {c}/({csq} + x²) dx, its position is x\u2080 = x(t\u2080). Let m be the slope of the curve y = arctan(x) at x = x\u2080. Evaluate ∫₀^m e^({k}x) dx.",
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

# c21: partial_frac → power → by_parts → volume
add("c21_partfrac_power_parts_vol",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Express {A}/((x − {p})(x − {q})) in partial fractions, and let b be the absolute value of the numerator corresponding to (x − {p}). A spindle is formed by rotating the region under a curve y = f(x) from x = 0 to x = b about the x-axis. In evaluating the integral ∫₀^b [f(x)]² dx using integration by parts, the boundary evaluation is ∫₀^b x^{n} dx, and the second term ∫ v du equals {iv}. Find the exact volume of the solid generated.",
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

# c22: partial_frac → reciprocal → exp → solve
add("c22_partfrac_recip_exp_solve",
    ["spec.integ.partial_frac", "spec.integ.reciprocal", "spec.integ.exp_linear", "spec.diffeq.solve_direct"],
    "Express {A}/((x − {p})(x − {q})) in partial fractions, and let M be the numerator corresponding to (x − {p}). With b = |M| + 1, let I = ∫₁^b (1/x) dx. A population grows according to dy/dt = e^t with y(0) = {y0}. Find the population y(I).",
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
    "Express {A}/((x − {p})(x − {q})) in partial fractions, and let M be the numerator corresponding to (x − {p}). With b = |M| + 1, let I = ∫₀^b x dx and J = ∫₁^(1+I) (1/x) dx. Evaluate ∫₀^J e^({k}x) dx.",
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

# c24: arcsin_eval → sin_sq → by_parts → volume
add("c24_arcsin_sinsq_parts_vol",
    ["spec.trig.arcsin_eval", "spec.integ.sin_sq", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "Let θ = arcsin({x_disp}). A solid of revolution is generated by rotating the region under a curve y = f(x) about the x-axis. In evaluating the integral ∫ [f(x)]² dx using integration by parts, the boundary evaluation is ∫₀^θ sin²(x) dx, and the remaining integral ∫ v du equals {iv}. If the solid's cross-sectional area is scaled by a factor of {C}, find the resulting volume of revolution.",
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

# c25: arccos_eval → cos_sq → by_parts → arcsin_val
add("c25_arccos_cossq_parts_arcsinval",
    ["spec.trig.arccos_eval", "spec.integ.cos_sq", "spec.integ.by_parts_step", "spec.deriv.arcsin_val"],
    "Let θ = arccos({x_disp}). In an integration by parts evaluation of an integral, the boundary evaluation is ∫₀^θ cos²(x) dx and the integral ∫ v du equals {iv}, yielding a net value J. Find the slope of the curve y = arcsin(x) at x = J/{scale}.",
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


# ---------------------------------------------------------------------
# DEPTH 5 (5 composites)
# ---------------------------------------------------------------------

# c26: arcsin_form → shm_eval → arctan_val → related_rate → solve_direct
add("c26_arcsinform_shm_arctanval_rate_solve",
    ["spec.integ.arcsin_form", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct"],
    "A particle moves in simple harmonic motion with displacement x(t) = {A} cos(t). At time t\u2080 = ∫₀^{b} 1/√({csq} − x²) dx, its displacement is x\u2080 = x(t\u2080). A tracking sensor measures viewing angle θ(x) = arctan(x). If the particle moves with horizontal speed dx/dt = {dxdt_d}, producing angular rate v = dθ/dt, and a signal accumulator satisfies dz/dt = v with z(0) = {z0}, find z({T}).",
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

# c27: arccos_eval → shm_eval → arctan_val → related_rate → power
add("c27_arccos_shm_arctanval_rate_power",
    ["spec.trig.arccos_eval", "spec.dynamics.shm_eval", "spec.deriv.arctan_val", "spec.deriv.related_rate", "spec.integ.power"],
    "An optical tracking system monitors a particle with harmonic displacement x(t) = {A} cos(t). At time t\u2080 = arccos({x_disp}), the particle has position x\u2080 = x(t\u2080). The sensor angle is θ(x) = arctan(x). If dx/dt = {dxdt_d}, producing angular speed v = |dθ/dt|, evaluate ∫₀^v x^{n} dx.",
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

# c28: partial_frac → power → volume → by_parts → solve
add("c28_partfrac_power_vol_parts_solve",
    ["spec.integ.partial_frac", "spec.integ.power", "spec.integ.volume_disk", "spec.integ.by_parts_step", "spec.diffeq.solve_direct"],
    "Express {A}/((x − {p})(x − {q})) in partial fractions, and let b be the numerator corresponding to (x − {p}). A spindle is formed by rotating the region under y = x^(n/2) from x = 0 to x = b about the x-axis, having volume V. In an integration by parts calculation, the boundary term is V and the second term ∫ v du equals {iv}, giving a net value J. A reservoir level satisfies dy/dx = g(x) with y(0) = {y0} and net inflow ∫ g(x) dx = J. Find the final level y.",
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

# c29: arctan_eval → sin_sq → by_parts → exp → solve
add("c29_arctan_sinsq_parts_exp_solve",
    ["spec.trig.arctan_eval", "spec.integ.sin_sq", "spec.integ.by_parts_step", "spec.integ.exp_linear", "spec.diffeq.solve_direct"],
    "Let θ = arctan({x_disp}). In an integration by parts calculation, the boundary evaluation is ∫₀^θ sin²(x) dx and the integral ∫ v du equals {iv}, yielding net value J. A quantity grows according to dy/dx = e^({k}x) with y(0) = {y0}. Find y(J).",
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

# c30: implicit → arcsin_val → rate → by_parts → volume
add("c30_implicit_arcsinval_rate_parts_vol",
    ["spec.deriv.implicit_grad", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.integ.by_parts_step", "spec.integ.volume_disk"],
    "A particle moves along the curve F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. The tangent slope is u = dy/dx, and the slope of the curve y = arcsin(x) at x = u is m. With horizontal speed dx/dt = {dxdt_d}, the resulting rate is v = |m·(dx/dt)|. In an integration by parts calculation, the boundary evaluation is v and the second integral ∫ v du equals {iv}, giving a net squared area J. Find the volume of the solid generated by rotating this region about the x-axis.",
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


# ---------------------------------------------------------------------
# DEPTH 6 (4 composites)
# ---------------------------------------------------------------------

# c31: margin → shm → arcsin_val → rate → solve → volume
add("c31_margin_shm_arcsinval_rate_solve_vol",
    ["spec.stats.margin_of_error", "spec.dynamics.shm_eval", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.diffeq.solve_direct", "spec.integ.volume_disk"],
    "A quality control test yields margin of error E for z = {z}, s = {s}, and n = {n_val}. A test oscillator has displacement x(t) = {A_disp} cos(t), reaching x\u2080 = x(E). Let m be the slope of the curve y = arcsin(x) at x = x\u2080. If dx/dt = {dxdt_d}, giving rate v = m·(dx/dt), and a reservoir level satisfies dw/dt = v with w(0) = {w0}, find w({T}). A solid of revolution is generated with squared profile integral equal to w({T}). Find its volume.",
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

# c32: arcsin_eval → shm → arcsin_form → sin_sq → volume → solve
add("c32_arcsin_shm_arcsinform_sinsq_vol_solve",
    ["spec.trig.arcsin_eval", "spec.dynamics.shm_eval", "spec.integ.arcsin_form", "spec.integ.sin_sq", "spec.integ.volume_disk", "spec.diffeq.solve_direct"],
    "A pendulum has release angle θ\u2081 = arcsin({x_disp}), reaching horizontal position x\u2080 = {A} cos(θ\u2081). The transit time across this distance is θ\u2082 = ∫₀^x\u2080 1/√({csq} − x²) dx. The energy transmission integral is I = ∫₀^θ\u2082 sin²(x) dx. A cylindrical containment has volume V = π I. If the initial storage level is y(0) = {y0}, find the final level y = y\u2080 + V.",
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

# c33: implicit → related_rate → power → by_parts → volume → solve
add("c33_implicit_rate_power_parts_vol_solve",
    ["spec.deriv.implicit_grad", "spec.deriv.related_rate", "spec.integ.power", "spec.integ.by_parts_step", "spec.integ.volume_disk", "spec.diffeq.solve_direct"],
    "A contour track follows F(x, y) = 0 where ∂F/∂x = {Fx} and ∂F/∂y = {Fy}. A vehicle moves with horizontal speed dx/dt = {dxdt_d}, giving vertical speed v = |(dy/dx)·(dx/dt)|. The work rate parameter is I = ∫₀^v x^{n} dx. In an integration by parts calculation, the boundary term is I and the second term ∫ v du equals {iv}, giving net cross-sectional integral J. A hydraulic chamber is formed by rotating this profile about the axis, giving volume V = π J. If the initial fluid volume is y(0) = {y0}, find the total volume y = y\u2080 + V.",
    {"Fx": {"type":"int","min":1,"max":4},
     "Fy": {"type":"int","min":2,"max":6},
     "r": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":2},
     "n": {"type":"int","min":1,"max":3},
     "iv": {"type":"int","min":1,"max":3},
     "y0": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.deriv.implicit_grad",
      "args":{"F_x":Q("Fx"),"F_y":Q("Fy")}},
     {"node_id":"n2","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n1"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k1","atom_id":"kernel.absolute_value","args":{"x":R("n2")}},
     {"node_id":"n3","atom_id":"spec.integ.power",
      "args":{"n":Q("n"),"a":L(0),"b":R("k1")}},
     {"node_id":"n4","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("n3"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"n5","atom_id":"spec.integ.volume_disk","args":{"integral_f_sq":R("n4")}},
     {"node_id":"n6","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("y0"),"integral_value":R("n5")}}],
    {"ref":"n6"},
    derive={"dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    ex_vars={"Fx":2,"Fy":3,"r":1,"s":1,"n":1,"iv":1,"y0":2})

# c34: arcsin_form → arcsin_val → related_rate → by_parts → reciprocal → solve
add("c34_arcsinform_arcsinval_rate_parts_recip_solve",
    ["spec.integ.arcsin_form", "spec.deriv.arcsin_val", "spec.deriv.related_rate", "spec.integ.by_parts_step", "spec.integ.reciprocal", "spec.diffeq.solve_direct"],
    "Let θ = ∫₀^{b} 1/√({csq} − x²) dx. The slope of the curve y = arcsin(x) at x = θ/{scale} is m. With horizontal speed dx/dt = {dxdt_d}, the resulting rate is v = |m·(dx/dt)|. In an integration by parts calculation, the boundary term is v and the second integral ∫ v du equals {iv}, yielding a net value J. A chemical concentration satisfies dc/dx = 1/x for x ≥ 1 with c(1) = {c0}. Find c(5 + J).",
    {"c": {"type":"int","min":3,"max":8},
     "b": {"type":"int","min":1,"max":5},
     "scale": {"type":"int","min":4,"max":8},
     "r": {"type":"int","min":1,"max":3},
     "s": {"type":"int","min":1,"max":2},
     "iv": {"type":"int","min":1,"max":3},
     "c0": {"type":"int","min":1,"max":5}},
    [{"node_id":"n1","atom_id":"spec.integ.arcsin_form",
      "args":{"c":Q("c"),"a":L(0),"b":Q("b")}},
     {"node_id":"k1","atom_id":"kernel.divide","args":{"x":R("n1"),"y":Q("scale")}},
     {"node_id":"n2","atom_id":"spec.deriv.arcsin_val","args":{"x":R("k1")}},
     {"node_id":"n3","atom_id":"spec.deriv.related_rate",
      "args":{"dy_dx":R("n2"),"dx_dt":Q("dxdt_val")}},
     {"node_id":"k2","atom_id":"kernel.absolute_value","args":{"x":R("n3")}},
     {"node_id":"n4","atom_id":"spec.integ.by_parts_step",
      "args":{"uv_upper":R("k2"),"uv_lower":L(0),"integral_vdu":Q("iv")}},
     {"node_id":"k3","atom_id":"kernel.add","args":{"x":L(5),"y":R("n4")}},
     {"node_id":"n5","atom_id":"spec.integ.reciprocal","args":{"a":L(1),"b":R("k3")}},
     {"node_id":"n6","atom_id":"spec.diffeq.solve_direct",
      "args":{"y0":Q("c0"),"integral_value":R("n5")}}],
    {"ref":"n6"},
    derive={"csq":"c * c","dxdt_val":"Fraction(r, s)","dxdt_d":"str(Fraction(r, s))"},
    constraints=["b < c"],
    ex_vars={"c":4,"b":2,"scale":5,"r":1,"s":1,"iv":1,"c0":2})


# =====================================================================
# VERIFY ALL AND WRITE
# =====================================================================

print(f"\nGenerated {len(composites)} composites and {len(graphs)} graphs")

# Depth check
depth_counts = {}
for g in graphs:
    d = check_program.knowledge_depth(g["nodes"], (g["return"]["ref"] if isinstance(g["return"], dict) else g["return"]))
    depth_counts[d] = depth_counts.get(d, 0) + 1
print(f"Depth distribution: {dict(sorted(depth_counts.items()))}")
target = {2:10, 3:8, 4:7, 5:5, 6:4}
assert depth_counts == target, f"Depth mismatch! Got {depth_counts}, want {target}"
print("✓ Exact depth distribution match!")

# Write files
with open(SM4 / "composite.jsonl", "w") as f:
    for c in composites:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"Written {len(composites)} composites to {SM4 / 'composite.jsonl'}")

with open(SM4 / "graphs.jsonl", "w") as f:
    for g in graphs:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
print(f"Written {len(graphs)} graphs to {SM4 / 'graphs.jsonl'}")

# Also update build_final_sm4.py by copying this script over
print("\nVerifying examples with check_program...")
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
