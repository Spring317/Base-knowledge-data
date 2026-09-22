from fractions import Fraction
import math

import sympy as sp

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernel import atom, REGISTRY, KIND, render


# ---------------------------------------------------------------------------
# Topic 1: Integration techniques  (ACMSM116–123)
# ---------------------------------------------------------------------------

@atom("spec.integ.sin_sq")
def integ_sin_sq(a, b):
    """∫_a^b sin²(x) dx  using  sin²x = (1 − cos 2x)/2."""
    a, b = sp.sympify(a), sp.sympify(b)
    return sp.simplify((b - a) / 2 - (sp.sin(2 * b) - sp.sin(2 * a)) / 4)


@atom("spec.integ.cos_sq")
def integ_cos_sq(a, b):
    """∫_a^b cos²(x) dx  using  cos²x = (1 + cos 2x)/2."""
    a, b = sp.sympify(a), sp.sympify(b)
    return sp.simplify((b - a) / 2 + (sp.sin(2 * b) - sp.sin(2 * a)) / 4)


@atom("spec.integ.sec_sq")
def integ_sec_sq(a, b):
    """∫_a^b sec²(x) dx = tan(b) − tan(a)."""
    a, b = sp.sympify(a), sp.sympify(b)
    return sp.simplify(sp.tan(b) - sp.tan(a))


@atom("spec.integ.power")
def integ_power(n, a, b):
    """∫_a^b x^n dx = (b^{n+1} − a^{n+1})/(n+1),  n ≠ −1."""
    if n == -1:
        raise ValueError("n = -1; use spec.integ.reciprocal")
    m = n + 1
    return (Fraction(b) ** m - Fraction(a) ** m) / m


@atom("spec.integ.reciprocal")
def integ_reciprocal(a, b):
    """∫_a^b (1/x) dx = ln(b) − ln(a)  for a, b > 0."""
    a, b = sp.sympify(a), sp.sympify(b)
    if a <= 0 or b <= 0:
        raise ValueError("limits must be positive")
    return sp.log(b) - sp.log(a)


@atom("spec.integ.exp_linear")
def integ_exp_linear(k, a, b):
    """∫_a^b e^{kx} dx = (e^{kb} − e^{ka})/k,  k ≠ 0."""
    if k == 0:
        raise ValueError("k must be nonzero")
    k, a, b = sp.sympify(k), sp.sympify(a), sp.sympify(b)
    return sp.simplify((sp.exp(k * b) - sp.exp(k * a)) / k)


# ---------------------------------------------------------------------------
# Inverse trigonometric functions  (ACMSM119)
# ---------------------------------------------------------------------------

@atom("spec.trig.arcsin_eval")
def arcsin_eval(x):
    """arcsin(x) for x ∈ [−1, 1]."""
    return sp.asin(sp.sympify(x))


@atom("spec.trig.arccos_eval")
def arccos_eval(x):
    """arccos(x) for x ∈ [−1, 1]."""
    return sp.acos(sp.sympify(x))


@atom("spec.trig.arctan_eval")
def arctan_eval(x):
    """arctan(x) for any real x."""
    return sp.atan(sp.sympify(x))


# ---------------------------------------------------------------------------
# Derivatives of inverse trig  (ACMSM120)
# ---------------------------------------------------------------------------

@atom("spec.deriv.arcsin_val")
def deriv_arcsin_val(x):
    """d/dx[arcsin(x)] = 1/√(1 − x²),  evaluated at x."""
    x = sp.sympify(x)
    return sp.simplify(1 / sp.sqrt(1 - x ** 2))


@atom("spec.deriv.arctan_val")
def deriv_arctan_val(x):
    """d/dx[arctan(x)] = 1/(1 + x²),  evaluated at x."""
    x = sp.sympify(x)
    return sp.simplify(sp.Rational(1) / (1 + x ** 2))


# ---------------------------------------------------------------------------
# Integrals producing inverse trig  (ACMSM121)
# ---------------------------------------------------------------------------

@atom("spec.integ.arcsin_form")
def integ_arcsin_form(c, a, b):
    """∫_a^b 1/√(c² − x²) dx = arcsin(b/c) − arcsin(a/c)."""
    c, a, b = sp.sympify(c), sp.sympify(a), sp.sympify(b)
    return sp.simplify(sp.asin(b / c) - sp.asin(a / c))


@atom("spec.integ.arctan_form")
def integ_arctan_form(c, a, b):
    """∫_a^b c/(c² + x²) dx = arctan(b/c) − arctan(a/c)."""
    c, a, b = sp.sympify(c), sp.sympify(a), sp.sympify(b)
    return sp.simplify(sp.atan(b / c) - sp.atan(a / c))


# ---------------------------------------------------------------------------
# Partial fractions  (ACMSM122)
# ---------------------------------------------------------------------------

@atom("spec.integ.partial_frac")
def integ_partial_frac(A, p, q):
    """Decompose A/((x−p)(x−q)) = M/(x−p) + N/(x−q).  Returns (M, N)."""
    if p == q:
        raise ValueError("p and q must be distinct")
    M = Fraction(A, p - q)
    N = Fraction(A, q - p)
    return (M, N)


# ---------------------------------------------------------------------------
# Integration by parts  (ACMSM123)
# ---------------------------------------------------------------------------

@atom("spec.integ.by_parts_step")
def integ_by_parts_step(uv_upper, uv_lower, integral_vdu):
    """∫_a^b u dv = [uv]_a^b − ∫_a^b v du
       = (uv_upper − uv_lower) − integral_vdu."""
    return (uv_upper - uv_lower) - integral_vdu


# ---------------------------------------------------------------------------
# Volume of revolution  (ACMSM125)
# ---------------------------------------------------------------------------

@atom("spec.integ.volume_disk")
def integ_volume_disk(integral_f_sq):
    """V = π ∫_a^b [f(x)]² dx = π · integral_f_sq."""
    return sp.pi * sp.sympify(integral_f_sq)


# ---------------------------------------------------------------------------
# Implicit differentiation  (ACMSM128)
# ---------------------------------------------------------------------------

@atom("spec.deriv.implicit_grad")
def deriv_implicit_grad(F_x, F_y):
    """dy/dx = −(∂F/∂x)/(∂F/∂y)  for the curve F(x,y) = 0."""
    if F_y == 0:
        raise ValueError("∂F/∂y must be nonzero")
    if isinstance(F_x, int) and isinstance(F_y, int):
        return Fraction(-F_x, F_y)
    return -sp.sympify(F_x) / sp.sympify(F_y)


# ---------------------------------------------------------------------------
# Related rates  (ACMSM129)
# ---------------------------------------------------------------------------

@atom("spec.deriv.related_rate")
def deriv_related_rate(dy_dx, dx_dt):
    """dy/dt = (dy/dx) · (dx/dt)  via the chain rule."""
    return dy_dx * dx_dt


# ---------------------------------------------------------------------------
# Differential equations  (ACMSM130)
# ---------------------------------------------------------------------------

@atom("spec.diffeq.solve_direct")
def diffeq_solve_direct(y0, integral_value):
    """Solve dy/dx = f(x), y(x₀) = y₀:
       y(x₁) = y₀ + ∫_{x₀}^{x₁} f(x) dx."""
    return y0 + integral_value


# ---------------------------------------------------------------------------
# Simple harmonic motion  (ACMSM136)
# ---------------------------------------------------------------------------

@atom("spec.dynamics.shm_eval")
def shm_eval(A, omega, phi, t):
    """x(t) = A cos(ωt + φ)."""
    A = sp.sympify(A)
    omega = sp.sympify(omega)
    phi = sp.sympify(phi)
    t = sp.sympify(t)
    return sp.simplify(A * sp.cos(omega * t + phi))


# ---------------------------------------------------------------------------
# Statistical inference  (ACMSM141 / ACMSM143)
# ---------------------------------------------------------------------------

@atom("spec.stats.margin_of_error")
def stats_margin_of_error(z, s, n):
    """Margin of error  E = z · s / √n.
       n must be a perfect square for an exact rational result."""
    sqrt_n = int(math.isqrt(n))
    if sqrt_n * sqrt_n != n:
        raise ValueError("n must be a perfect square for exact output")
    return Fraction(z * s, sqrt_n)
