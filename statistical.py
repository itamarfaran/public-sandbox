import numpy as np
from scipy import stats, optimize


def calculate_p_B(
    n: float,
    p: float,
    xi: float,
    alpha: float,
    power: float,
) -> float:
    """
    n: total sample size
    p: baseline (control) conversion rate
    xi: fraction of n allocated to the treatment/variant group (n_B = n*xi, n_A = n*(1-xi))
    alpha: significance level (two-sided)
    power: target statistical power
    Returns p_B, the treatment rate at which MDE is achieved (unpooled-variance z approximation;
    matches a test that uses the unpooled standard error, not the pooled one under H0).
    Returns nan if no p_B <= 1 is detectable with this sample size.
    """
    z_sum = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    z_sq = z_sum**2

    a = n * xi + z_sq
    b = -(2 * n * xi * p + z_sq)
    c = n * xi * p**2 - z_sq * xi * p * (1 - p) / (1 - xi)

    discriminant = b**2 - 4 * a * c
    p_B = (-b + np.sqrt(discriminant)) / (2 * a)

    if p_B > 1:
        return float("nan")
    return p_B


def calculate_mu_B(
    n: float,
    mu: float,
    sigma: float,
    xi: float,
    alpha: float,
    power: float,
) -> float:
    """
    n: total sample size
    mu: baseline (control) mean
    sigma: common standard deviation of both groups (equal-variance assumption)
    xi: fraction of n allocated to the treatment/variant group (n_B = n*xi, n_A = n*(1-xi))
    alpha: significance level (two-sided)
    power: target statistical power
    Returns mu_B, the treatment mean at which MDE is achieved (large-sample z approximation;
    somewhat anti-conservative for small groups, where a t-based calculation is exact).
    """
    z_sum = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    se = sigma * np.sqrt(1 / (n * xi) + 1 / (n * (1 - xi)))
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
