"""Numerical checks of the equivalent forms in the paper, with exact transforms (float64).

mu(S) = fhat(S)^2; up_t(T) = sum_{U >= T} mu(U) t^{|U - T|}; m_t(T) = t (d/dt) up_t(T) / up_t(T);
A = E_mu log2(1/up_1), B = E_mu log2(up_1/mu), E_i = H(1[i in U] | U - i), L the per-element erasure loss."""
import math
import numpy as np

LN2 = math.log(2)


def wht(v):
    a = v.astype(float).copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        x, y = a[:, :h].copy(), a[:, h:].copy()
        a[:, :h], a[:, h:] = x + y, x - y
        a = a.reshape(-1)
        h *= 2
    return a / len(a)


def up_and_deriv(mu, n, t):
    """superset sums with weight t per extra element, and their t-derivative."""
    a, da = mu.copy(), np.zeros_like(mu)
    idx = np.arange(len(mu))
    for k in range(n):
        lo = idx[((idx >> k) & 1) == 0]
        hi = lo | (1 << k)
        a_hi, da_hi = a[hi].copy(), da[hi].copy()
        da[lo] = da[lo] + a_hi + t * da_hi
        a[lo] = a[lo] + t * a_hi
    return a, da


def popcount(n):
    return np.array([bin(j).count("1") for j in range(1 << n)])


def eta(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x > 1e-300
    out[m] = -x[m] * np.log2(x[m])
    return out


def checks(f, n, rng):
    mu = wht(f) ** 2
    pc = popcount(n)
    supp = mu > 1e-30
    H = float(eta(mu).sum())
    I = float((pc * mu).sum())
    Var = 1 - mu[0]
    up1, _ = up_and_deriv(mu, n, 1.0)
    A = float((mu[supp] * np.log2(1 / up1[supp])).sum())
    B = float((mu[supp] * np.log2(up1[supp] / mu[supp])).sum())
    r = {}
    nz = supp & (pc > 0)
    r["gran"] = float(np.max(mu[nz] / (4.0 ** (pc[nz] - 1) * up1[nz] ** 2))) if nz.any() else 0.0
    r["split"] = H - (2 * B + 2 * I - 2 * Var)          # <= 0
    r["sum"] = abs(H - A - B)
    worst_b = -1e9
    for t in (0.25, 0.5, 0.75, 0.9):
        upt, _ = up_and_deriv(mu, n, t)
        At = float((mu[supp] * np.log2(1 / upt[supp])).sum())
        worst_b = max(worst_b, H - At - I * math.log2(1 / (1 - t)), At - H)
    r["thin"] = worst_b                                   # <= 0
    xs, ws = np.polynomial.legendre.leggauss(80)
    ts, wt = 0.5 * (xs + 1), 0.5 * ws
    integ = 0.0
    prev = None
    mono = True
    for t, w in sorted(zip(ts, wt)):
        upt, dupt = up_and_deriv(mu, n, t)
        mt = t * dupt[supp] / upt[supp]
        if prev is not None and np.any(mt < prev - 1e-12):
            mono = False
        prev = mt
        integ += w * float((mu[supp] * mt).sum()) / t
    r["tilt"] = abs(B - integ / LN2)
    _u1, _d1 = up_and_deriv(mu, n, 1.0)
    M1 = float((mu[supp] * _d1[supp] / _u1[supp]).sum())
    r["m1"] = H - (4 * I - 2 * Var + 2 * M1)              # <= 0
    r["mono"] = mono
    idx = np.arange(1 << n)
    sumE, IL = 0.0, 0.0
    for i in range(n):
        lo = idx[((idx >> i) & 1) == 0]
        p, q = mu[lo], mu[lo | (1 << i)]
        sumE += float((eta(p) + eta(q) - eta(p + q)).sum())
        m = q > 1e-300
        IL += float((q[m] * np.log2((p[m] + q[m]) / q[m])).sum())
    r["Lbracket"] = max(IL - sumE, sumE - IL - I / LN2)   # <= 0
    # constant predictor p = 1/2: K = (1/I) sum (mu(T)+mu(T+i))/2 = n/(2I); bound L <= log2 K + 1
    r["pred"] = IL / I - (math.log2(n / (2 * I)) + 1)     # <= 0
    R = int(rng.integers(0, 1 << n))
    g = f * np.array([(-1) ** bin(j & R).count("1") for j in range(1 << n)])
    mug = wht(g) ** 2
    sumEg = 0.0
    for i in range(n):
        lo = idx[((idx >> i) & 1) == 0]
        p, q = mug[lo], mug[lo | (1 << i)]
        sumEg += float((eta(p) + eta(q) - eta(p + q)).sum())
    r["transl"] = abs(sumE - sumEg)
    return r, H, I, sumE


def tribes(w, s):
    n = w * s
    f = np.ones(1 << n, dtype=int)
    for j in range(1 << n):
        if any(all((j >> (b * w + k)) & 1 for k in range(w)) for b in range(s)):
            f[j] = -1
    return f, n


rng = np.random.default_rng(11)
fams = []
for n in range(3, 10):
    for bias in (0.5, 0.15, 0.03):
        for _ in range(6):
            f = np.where(rng.random(1 << n) < bias, -1, 1)
            if not np.all(f == f[0]):
                fams.append((f"rand n={n} p={bias}", f, n))
and8 = np.ones(256, dtype=int); and8[255] = -1
fams.append(("AND_8", and8, 8))
for w, s in ((2, 4), (3, 3)):
    f, n = tribes(w, s)
    fams.append((f"tribes {w}x{s}", f, n))
maj = np.array([1 if bin(j).count("1") < 5 else -1 for j in range(512)])
fams.append(("MAJ_9", maj, 9))

agg = {}
for name, f, n in fams:
    r, H, I, sE = checks(f, n, rng)
    for k, v in r.items():
        if k == "mono":
            agg[k] = agg.get(k, True) and v
        else:
            agg[k] = max(agg.get(k, -1e18), v)
    if not name.startswith("rand"):
        print(f"{name:12s} H={H:.6f} I={I:.6f} sumE/I={sE / I:.4f} " +
              " ".join(f"{k}={v:.3g}" if not isinstance(v, bool) else f"{k}={v}" for k, v in r.items()))
print(f"functions checked: {len(fams)}")
print("max over all (should be: gran <= 1, split/thin/m1/Lbracket/pred <= 0, sum/tilt/transl ~ 0, mono True):")
for k, v in agg.items():
    print(f"  {k}: {v}")
