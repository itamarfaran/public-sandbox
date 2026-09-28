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
    Returns p_B, the treatment rate at which MDE is achieved (unpooled-variance approximation).
    """
    dist = stats.norm(mu=0, sigma=1)
    z_diff = (dist.ppf(1 - alpha / 2) - dist.ppf(1 - power)) ** 2

    a = n * xi + z_diff
    b = -(2 * n * xi * p + z_diff)
    c = n * xi * p**2 - z_diff * xi * p * (1 - p) / (1 - xi)

    discriminant = b**2 - 4 * a * c
    return (-b + np.sqrt(discriminant)) / (2 * a)


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
    Returns mu_B, the treatment mean at which MDE is achieved (large-sample z approximation).
    """
    dist = stats.norm(mu=0, sigma=1)
    z_sum = dist.ppf(1 - alpha / 2) + dist.ppf(power)
    se = sigma * np.sqrt(1 / (n * xi) + 1 / (n * (1 - xi)))
    return mu + z_sum * se


def dunnett_d(
    *sizes: float,
    alpha: float = 0.05,
    two_sided: bool = True,
    eps: float = 1e-5,
) -> float:
    """
    Large-sample Dunnett critical value for comparing each treatment to one control.
    *sizes: group sizes (or fractions); the FIRST is the control, the rest are treatments.
            e.g. dunnett_d(300, 100, 200, 50) or dunnett_d([300, 100, 200, 50])
    alpha: family-wise significance level
    two_sided: reject when |Z_i| > d (True) or Z_i > d (False)
    """
    sizes = np.ravel(np.array(sizes, dtype=float))
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
    mvn = stats.multivariate_normal(mean=np.zeros(k), cov=corr, abseps=eps, releps=eps)

    def excess(d):
        lower = np.full(k, -d) if two_sided else np.full(k, -np.inf)
        return mvn.cdf(np.full(k, d), lower_limit=lower) - (1 - alpha)

    return optimize.brentq(
        excess,
        stats.norm.ppf(1 - a) - 0.01,
        stats.norm.ppf(1 - a / k) + 0.01,
        xtol=1e-6,
    )
