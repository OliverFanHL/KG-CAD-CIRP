"""Conservative relevance bounds and fixed-cutoff retrieval metrics."""
import numpy as np
TOL = 1e-12
SCENARIOS = ("nominal", "conditional_zero", "conditional_two")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def prepare_limits(lower, upper, cutoffs=(5, 10, 20), ap_cutoff=20):
    """Cache cohort-wide denominator extrema, independent of retrieval method."""
    lo, hi = np.asarray(lower), np.asarray(upper)
    require(lo.ndim == hi.ndim == 1 and lo.shape == hi.shape, "Grade-bound vector shape mismatch")
    require(lo.dtype.kind in "iu" and hi.dtype.kind in "iu"
            and np.all((0 <= lo) & (lo <= hi) & (hi <= 2)), "Invalid grade endpoints")
    require(len(set(cutoffs)) == len(cutoffs) and all(type(k) is int and k > 0 for k in cutoffs)
            and type(ap_cutoff) is int and ap_cutoff > 0, "Invalid cutoff")
    discount = 1. / np.log2(np.arange(max(cutoffs), dtype=float) + 2.)
    ideal_lo = np.cumsum((np.exp2(np.sort(lo)[::-1][:max(cutoffs)]) - 1.) * discount[:min(max(cutoffs), len(lo))])
    ideal_hi = np.cumsum((np.exp2(np.sort(hi)[::-1][:max(cutoffs)]) - 1.) * discount[:min(max(cutoffs), len(hi))])

    def at(values, k):
        return float(values[min(k, len(values)) - 1]) if len(values) else 0.

    return dict(lower=lo.astype(np.int8), upper=hi.astype(np.int8), cutoffs=tuple(cutoffs), ap_cutoff=ap_cutoff,
                discount=discount, rlo=int(np.count_nonzero(lo)), rhi=int(np.count_nonzero(hi)),
                idcg={k: (at(ideal_lo, k), at(ideal_hi, k)) for k in cutoffs})

def _ratio(nlo, nhi, dlo, dhi, unit_interval=True):
    """Include denominator-zero metric=0; never drop a zero-relevance query."""
    if dhi == 0:
        return 0., 0.
    lower = nlo / dhi
    upper = nhi / dlo if dlo else (1. if nhi > 0 else 0.)
    return float(lower), float(min(1., upper) if unit_interval else upper)

def rank_bounds(order, prepared):
    """Vectorized monotone numerators over separately bounded denominators."""
    order = np.asarray(order)
    require(order.ndim == 1 and order.dtype.kind in "iu", "Ranking must be an integer index vector")
    require(len(order) == len(np.unique(order)) and np.all((order >= 0) & (order < len(prepared["lower"]))),
            "Duplicate or out-of-cohort ranking index")
    lo, hi = prepared["lower"][order], prepared["upper"][order]
    hit_lo, hit_hi = (lo > 0).astype(np.int64), (hi > 0).astype(np.int64)
    hits_lo, hits_hi = np.cumsum(hit_lo), np.cumsum(hit_hi)
    strong_lo, strong_hi = np.cumsum(lo == 2), np.cumsum(hi == 2)
    maximum = max(prepared["cutoffs"])
    weights = prepared["discount"][:min(maximum, len(order))]
    gains_lo = np.cumsum((np.exp2(lo[:maximum]) - 1.) * weights)
    gains_hi = np.cumsum((np.exp2(hi[:maximum]) - 1.) * weights)

    def at(values, k):
        return float(values[min(k, len(values)) - 1]) if len(values) else 0.

    bounds = {}
    for k in prepared["cutoffs"]:
        idlo, idhi = prepared["idcg"][k]
        bounds[f"ndcg@{k}"] = _ratio(at(gains_lo, k), at(gains_hi, k), idlo, idhi)
        bounds[f"precision@{k}"] = at(hits_lo, k) / k, at(hits_hi, k) / k
        bounds[f"strong_precision@{k}"] = at(strong_lo, k) / k, at(strong_hi, k) / k
        # Relevance count is integer: if it can be zero, its minimum positive
        # value is at least 1; the upper ratio is capped at 1.
        bounds[f"recall@{k}"] = _ratio(at(hits_lo, k), at(hits_hi, k),
                                       max(1, prepared["rlo"]), prepared["rhi"])
    ranks = np.arange(1, len(order) + 1, dtype=float)
    numerator_lo = np.cumsum(hit_lo * hits_lo / ranks)
    numerator_hi = np.cumsum(hit_hi * hits_hi / ranks)
    full_nlo = float(numerator_lo[-1]) if len(order) else 0.
    full_nhi = float(numerator_hi[-1]) if len(order) else 0.
    bounds["ap"] = _ratio(full_nlo, full_nhi, max(1, prepared["rlo"]), prepared["rhi"])
    apk = prepared["ap_cutoff"]
    bounds[f"ap@{apk}"] = _ratio(at(numerator_lo, apk), at(numerator_hi, apk),
                                   max(1, min(apk, prepared["rlo"])), min(apk, prepared["rhi"]))
    for name, (lower, upper) in bounds.items():
        require(-TOL <= lower <= upper + TOL and upper <= 1 + TOL, f"Invalid envelope: {name}")
    return bounds

def grade_limits(raw, verdicts, scenario="nominal"):
    """Endpoint image of the unchanged monotone raw/conditional/size transforms.

    A null retains all raw grades 0..2. Conditional transforms may create gaps
    such as {0,2}; endpoints suffice for monotone numerator/denominator extrema
    and do not assert that a missing intermediate grade is admissible.
    """
    require(scenario in SCENARIOS, "Unknown conditional scenario")
    raw = np.asarray(raw, dtype=int)
    require(np.all(np.isin(raw, [-1, 0, 1, 2])), "Invalid raw grade vector")
    require(len(raw) == len(verdicts) and all(v in {"pass", "not_requested", "unknown", "fail"} for v in verdicts),
            "Invalid reference verdict vector")
    lo, hi = np.where(raw < 0, 0, raw), np.where(raw < 0, 2, raw)
    if scenario != "nominal":
        value = 0 if scenario == "conditional_zero" else 2
        lo, hi = np.where(lo == 1, value, lo), np.where(hi == 1, value, hi)
    verdicts = np.asarray(verdicts)
    unknown, fail = verdicts == "unknown", verdicts == "fail"
    lo[unknown], hi[unknown] = np.minimum(lo[unknown], 1), np.minimum(hi[unknown], 1)
    lo[fail], hi[fail] = 0, 0
    return lo, hi
