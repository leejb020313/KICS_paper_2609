"""Calibration methods: standard baselines + NN-PPI reproduction + our extension.

All methods share the same interface: fit on (raw_scores, labels) from the
calibration set, then transform raw scores on new instances into calibrated
scores in roughly [0, 1].

Ours = NN-PPI base + up to three independently toggleable fixes, so each can
be ablated on its own:
  - similarity weighting: weight neighbor residuals by cosine similarity
    instead of uniform 1/k averaging (motivated by arXiv 2606.31577, which
    found *naive* cosine weighting alone doesn't help -- so we softmax-sharpen
    it with a temperature rather than using raw similarity as the weight).
  - range clipping: clip the final calibrated score into [0, 1].
  - variance fix: weighted residual variance instead of the paper's
    unweighted sigma_res^2 / |S_i|, to test whether it improves empirical CI
    coverage.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def temperature_scale_fit(raw_scores: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single temperature T minimizing NLL on logit(raw_score)/T -> label."""
    eps = 1e-6
    logits = np.log(np.clip(raw_scores, eps, 1 - eps) / np.clip(1 - raw_scores, eps, 1 - eps))

    def nll(T):
        p = 1 / (1 + np.exp(-logits / T))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

    # simple 1D grid + local refine (no scipy dependency assumed)
    grid = np.linspace(0.05, 5.0, 200)
    losses = [nll(t) for t in grid]
    best_T = grid[int(np.argmin(losses))]
    return best_T


def temperature_scale_apply(raw_scores: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    logits = np.log(np.clip(raw_scores, eps, 1 - eps) / np.clip(1 - raw_scores, eps, 1 - eps))
    return 1 / (1 + np.exp(-logits / T))


def platt_scale_fit(raw_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression()
    lr.fit(raw_scores.reshape(-1, 1), labels)
    return lr


def platt_scale_apply(raw_scores: np.ndarray, model: LogisticRegression) -> np.ndarray:
    return model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]


def isotonic_fit(raw_scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, labels)
    return iso


def isotonic_apply(raw_scores: np.ndarray, model: IsotonicRegression) -> np.ndarray:
    return model.predict(raw_scores)


@dataclass
class NNPPIConfig:
    k: int = 5
    similarity_weighted: bool = False
    softmax_temperature: float = 0.1  # only used if similarity_weighted
    clip_range: bool = False
    weighted_variance: bool = False
    variance_shrinkage: bool = False
    shrinkage_prior_strength: float = 5.0  # m in ESS/(ESS+m); fit-free, chosen a priori
    global_residual_var: float = None  # must be set from calib-only data when variance_shrinkage=True


@dataclass
class NNPPIResult:
    theta: np.ndarray          # calibrated scores
    ci_half_width: np.ndarray  # z * sqrt(var)
    sqrt_var: np.ndarray       # sqrt(var), i.e. ci_half_width / z -- exposed so
                                # a conformal scale factor can replace z directly
    out_of_range_rate: float


def nn_ppi_apply(
    test_raw_scores: np.ndarray,
    test_embeddings: np.ndarray,
    calib_raw_scores: np.ndarray,
    calib_embeddings: np.ndarray,
    calib_labels: np.ndarray,
    config: NNPPIConfig,
    alpha: float = 0.05,
) -> NNPPIResult:
    """Vectorized NN-PPI (and our ablations) over L2-normalized embeddings."""
    # cosine similarity via dot product (embeddings must be L2-normalized)
    sims = test_embeddings @ calib_embeddings.T  # (n_test, n_calib)
    k = config.k
    neighbor_idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]

    n_test = test_raw_scores.shape[0]
    theta = np.empty(n_test)
    var = np.empty(n_test)
    z = 1.959963985  # z_{0.975} for 95% CI

    for i in range(n_test):
        idx = neighbor_idx[i]
        neighbor_residuals = calib_labels[idx] - calib_raw_scores[idx]  # (k,)

        if config.similarity_weighted:
            neighbor_sims = sims[i, idx]
            # softmax-sharpen the similarities into weights (raw cosine
            # weighting alone was shown insufficient in arXiv 2606.31577)
            w = np.exp(neighbor_sims / config.softmax_temperature)
            w = w / w.sum()
        else:
            w = np.full(k, 1.0 / k)

        r_bar = float(np.sum(w * neighbor_residuals))
        theta_i = test_raw_scores[i] + r_bar

        if config.weighted_variance or config.variance_shrinkage:
            # weighted residual variance (effective-sample-size corrected)
            mean = r_bar
            weighted_var = np.sum(w * (neighbor_residuals - mean) ** 2)
            ess = 1.0 / np.sum(w ** 2)  # effective sample size
            local_var_i = weighted_var / max(ess, 1e-6)
        else:
            ess = k
            local_var_i = np.var(neighbor_residuals) / k

        if config.variance_shrinkage:
            assert config.global_residual_var is not None, "set global_residual_var from calib-only data"
            lam = ess / (ess + config.shrinkage_prior_strength)
            var_i = lam * local_var_i + (1 - lam) * config.global_residual_var
        else:
            var_i = local_var_i

        theta[i] = theta_i
        var[i] = var_i

    out_of_range_rate = float(np.mean((theta < 0) | (theta > 1)))
    if config.clip_range:
        theta = np.clip(theta, 0.0, 1.0)

    sqrt_var = np.sqrt(np.maximum(var, 0.0))
    ci_half_width = z * sqrt_var
    return NNPPIResult(theta=theta, ci_half_width=ci_half_width, sqrt_var=sqrt_var,
                        out_of_range_rate=out_of_range_rate)


def shrinkage_m_fit(
    holdout_raw_scores: np.ndarray,
    holdout_embeddings: np.ndarray,
    holdout_labels: np.ndarray,
    neighbor_raw_scores: np.ndarray,
    neighbor_embeddings: np.ndarray,
    neighbor_labels: np.ndarray,
    base_config: NNPPIConfig,
    global_residual_var: float,
    candidates=(1.0, 2.0, 5.0, 10.0, 20.0),
    target_coverage: float = 0.95,
) -> float:
    """Pick the shrinkage prior strength m from `candidates` using ONLY a
    held-out slice of the calibration set (never the test set) -- same
    discipline as conformal_scale_fit. Chooses the smallest m (narrowest
    intervals) whose holdout coverage reaches target_coverage; if none reach
    it, returns the m with the highest holdout coverage.
    """
    from nnppi.evaluate import ci_coverage as _ci_coverage
    best_m, best_cov, best_width = None, -1.0, float("inf")
    for m in candidates:
        cfg = NNPPIConfig(k=base_config.k, similarity_weighted=base_config.similarity_weighted,
                           clip_range=base_config.clip_range, variance_shrinkage=True,
                           shrinkage_prior_strength=m, global_residual_var=global_residual_var)
        res = nn_ppi_apply(holdout_raw_scores, holdout_embeddings,
                            neighbor_raw_scores, neighbor_embeddings, neighbor_labels, cfg)
        cov = _ci_coverage(holdout_labels, res.theta, res.ci_half_width)
        width = float(np.mean(res.ci_half_width))
        if cov >= target_coverage and width < best_width:
            best_m, best_cov, best_width = m, cov, width
        elif best_m is None and cov > best_cov:
            best_m, best_cov, best_width = m, cov, width
    return best_m


def conformal_scale_fit(
    holdout_raw_scores: np.ndarray,
    holdout_embeddings: np.ndarray,
    holdout_labels: np.ndarray,
    neighbor_raw_scores: np.ndarray,
    neighbor_embeddings: np.ndarray,
    neighbor_labels: np.ndarray,
    config: NNPPIConfig,
    target_coverage: float = 0.95,
) -> float:
    """Split-conformal scale factor, estimated ONLY on a held-out slice of the
    calibration set (never the test set). Replaces the parametric z=1.96 with
    an empirically-fit multiplier q such that theta +/- q*sqrt_var achieves
    close to target_coverage on the holdout slice. `neighbor_*` must be
    disjoint from `holdout_*` (e.g. an 80/20 split of the original calib set).
    """
    res = nn_ppi_apply(holdout_raw_scores, holdout_embeddings,
                        neighbor_raw_scores, neighbor_embeddings, neighbor_labels,
                        config)
    nonconformity = np.abs(holdout_labels - res.theta) / np.maximum(res.sqrt_var, 1e-6)
    q = float(np.quantile(nonconformity, target_coverage))
    return q
