from collections.abc import Sequence

import numpy as np
from scipy import stats, optimize


def _d_and_xi(xi: float | Sequence[float], alpha: float) -> tuple[float, float, float]:
    """
    Returns (d, xi0, xi_t): the two-sided Dunnett critical value, the control fraction of n,
    and the treatment fraction of n (for k > 1, the smallest treatment group, whose MDE is
    the largest).
    """
    if isinstance(xi, (int, float, np.number)):
        d = dunnett_d(1 - xi, xi, alpha=alpha)
        return d, 1 - xi, xi
    else:
        d = dunnett_d(*xi, alpha=alpha)
        sum_xi = sum(xi)
        return d, xi[0] / sum_xi, min(xi[1:]) / sum_xi


def calculate_p_B(
    n: float,
    p: float,
    xi: float | Sequence[float],
    alpha: float,
    power: float,
) -> float:
    """
    n: total sample size
    p: baseline (control) conversion rate
    xi: float for an A/B test: fraction of n allocated to the treatment group
        (n_B = n*xi, n_A = n*(1-xi)); or a sequence of all group sizes (or weights), control
        first, e.g. [400, 200, 200, 200]; normalized so the groups sum to n
    alpha: family-wise significance level (two-sided; Dunnett-adjusted when k > 1)
    power: target per-arm statistical power
    Returns p_B, the treatment rate at which MDE is achieved (unpooled-variance z approximation;
    matches a test that uses the unpooled standard error, not the pooled one under H0).
    When xi is a sequence, returns p_B for the smallest treatment group (the largest MDE).
    Returns nan if no p_B <= 1 is detectable with this sample size.
    """
    d, xi0, xi_t = _d_and_xi(xi, alpha)
    z_sum = d + stats.norm.ppf(power)
    z_sq = z_sum**2

    a = n * xi_t + z_sq
    b = -(2 * n * xi_t * p + z_sq)
    c = n * xi_t * p**2 - z_sq * xi_t * p * (1 - p) / xi0

    discriminant = b**2 - 4 * a * c
    p_B = (-b + np.sqrt(discriminant)) / (2 * a)

    return p_B if p_B <= 1 else float("nan")


def calculate_mu_B(
    n: float,
    mu: float,
    sigma: float,
    xi: float | Sequence[float],
    alpha: float,
    power: float,
) -> float:
    """
    n: total sample size
    mu: baseline (control) mean
    sigma: common standard deviation of all groups (equal-variance assumption)
    xi: float for an A/B test: fraction of n allocated to the treatment group
        (n_B = n*xi, n_A = n*(1-xi)); or a sequence of all group sizes (or weights), control
        first, e.g. [400, 200, 200, 200]; normalized so the groups sum to n
    alpha: family-wise significance level (two-sided; Dunnett-adjusted when k > 1)
    power: target per-arm statistical power
    Returns mu_B, the treatment mean at which MDE is achieved (large-sample z approximation;
    somewhat anti-conservative for small groups, where a t-based calculation is exact).
    When xi is a sequence, returns mu_B for the smallest treatment group (the largest MDE).
    """
    d, xi0, xi_t = _d_and_xi(xi, alpha)
    z_sum = d + stats.norm.ppf(power)
    se = sigma * np.sqrt(1 / (n * xi_t) + 1 / (n * xi0))
    return mu + z_sum * se


def dunnett_d(
    *sizes: float,
    alpha: float = 0.05,
    two_sided: bool = True,
    eps: float = 1e-5,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """
    Large-sample Dunnett critical value for comparing each treatment to one control.
    *sizes: group sizes (or fractions); the FIRST is the control, the rest are treatments.
    alpha: family-wise significance level
    two_sided: reject when |Z_i| > d (True) or Z_i > d (False)
    eps: absolute/relative error tolerance of the multivariate normal CDF
    seed: seed for the randomized MVN CDF integration, so repeated calls return the same
          value; None gives run-to-run variation of roughly eps in the CDF
    Uses the z approximation (known variances), so it is somewhat anti-conservative for
    small groups compared with the exact t-based Dunnett value.
    """
    sizes = np.asarray(sizes, dtype=float)
    if sizes.ndim != 1:
        raise ValueError("Group sizes must be a flat sequence of numbers")
    if sizes.size < 2:
        raise ValueError("Need a control size plus at least one treatment size")
    if np.any(sizes <= 0):
        raise ValueError("All group sizes must be positive")
    n0, ni = sizes[0], sizes[1:]
    k = len(ni)
    a = alpha / 2 if two_sided else alpha
    if k == 1:
        return stats.norm.ppf(1 - a)

    # corr(Z_i, Z_j) = sqrt(n_i n_j / ((n_i + n0)(n_j + n0)))
    r = np.sqrt(ni / (ni + n0))
    corr = np.outer(r, r)
    np.fill_diagonal(corr, 1.0)
    mvn = stats.multivariate_normal(
        mean=np.zeros(k),
        cov=corr,
        abseps=eps,
        releps=eps,
        seed=seed,
    )

    def excess(d):
        lower = np.full(k, -d) if two_sided else np.full(k, -np.inf)
        return mvn.cdf(np.full(k, d), lower_limit=lower) - (1 - alpha)

    return optimize.brentq(
        excess,
        stats.norm.ppf(1 - a) - 0.01,
        stats.norm.ppf(1 - a / k) + 0.01,
        xtol=1e-6,
    )
